"""
F-02: Streamlit-free port of app.py's retrieval + Moneyball scoring pipeline.

Ported from the current (post-LanceDB) app.py, not the reference repo's stale
SQLite/FAISS pipeline_core.py. Per docs/PLAN.md's F-01..F-06 risk note, this is
a fresh copy — app.py is intentionally left untouched until the F-06 cutover
is explicitly approved, so keep this file in sync with app.py by hand when the
pipeline changes, don't assume the two auto-drift together.

Decoupling from app.py's Streamlit version:
  - @st.cache_resource  -> @functools.lru_cache(maxsize=1) (same one-time-load
    semantics for these zero-arg model/connection loaders)
  - st.secrets           -> os.getenv only (backend has no Streamlit runtime)
  - st.session_state     -> plain module-level dict (_STATE) — correct here
    since a backend process is a singleton, not a per-user Streamlit session
  - st.write/info/warning/progress -> emit(msg, level) callback / on_progress
    callback, same seam select_embedding_candidates already used
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import hashlib
import json
import logging
import math
import re
import textwrap
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    from google import genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

try:
    from query_intelligence import analyse_query
except ImportError:
    analyse_query = None  # type: ignore

# =========================
# Constants
# =========================

MIN_FOR_PREDICTION = 20

DEFAULT_MONEYBALL_WEIGHTS = {
    "weight_fame": 0.84,
    "weight_hype": 0.0,
    "weight_sniper": 0.0,
    "weight_utility": 0.16,
}

# QIL quality_modifier -> LanceDB pre-filter (docs/qil-improvements-planner.md B1, Job 1).
QUALITY_LANCEDB_FILTERS = {
    "recent": "year >= 2023",
    "influential": "citation_count >= 50",
    "emerging": "year >= 2023 AND citation_count < 20",
    "classic": "year <= 2018",
}

# QIL quality_modifier -> Moneyball weight override (B1, Job 2). "recent" uses an
# additive recency bonus instead (RECENCY_BONUS_MAX in predict_citations_direct).
QUALITY_MONEYBALL_WEIGHTS = {
    "influential": {"weight_fame": 0.95, "weight_hype": 0.0, "weight_sniper": 0.0, "weight_utility": 0.05},
    "emerging": {"weight_fame": 0.30, "weight_hype": 0.0, "weight_sniper": 0.0, "weight_utility": 0.70},
    "classic": {"weight_fame": 0.90, "weight_hype": 0.0, "weight_sniper": 0.0, "weight_utility": 0.10},
}

PRIMARY_THRESHOLD: float = 0.55
SECONDARY_THRESHOLD: float = 0.25
_CE_MAX_ABSTRACT_CHARS: int = 1500
DB_FETCH_LIMIT = 20_000
RECENCY_BONUS_MAX = 15.0

# Stage 1 RRF fusion constant. 60 is the common empirical default, but
# literature on RRF k-tuning shows k=30-46 often outperforms it by giving
# top-ranked results more relative weight — see docs/relevance-strategy-comparison.md
# Approach A1. Lower = more weight to top ranks; higher = more consensus-seeking.
RRF_K = 45

CONFERENCE_KEYWORDS = [
    "EMNLP", "ACL", "NAACL", "EACL",
    "NeurIPS", "ICLR", "ICML",
    "CVPR", "ECCV",
    "ICASSP", "AAAI", "AISTATS",
]

JOURNAL_KEYWORDS = [
    "Nature", "Science",
    "JMLR", "Journal of Machine Learning Research",
    "TPAMI", "IEEE Transactions on Pattern Analysis",
    "Artificial Intelligence Journal",
    "IJCV", "International Journal of Computer Vision",
    "Nature Machine Intelligence", "Nature Communications",
]

NEGATIVE_VENUE_SIGNALS = ["submitted to", "under review", "preprint"]

ARXIV_CATEGORIES: Dict[str, List[str]] = {
    "Computer Science": [
        "cs.AI", "cs.LG", "cs.HC", "cs.CL", "cs.CV", "cs.RO", "cs.IR", "cs.NE", "cs.SE",
        "cs.CR", "cs.DS", "cs.DB", "cs.SI", "cs.MM", "cs.IT", "cs.PF", "cs.MA",
    ],
}

ARXIV_CODE_TO_NAME = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.HC": "Human-Computer Interaction",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.RO": "Robotics",
    "cs.IR": "Information Retrieval",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.SE": "Software Engineering",
    "cs.CR": "Cryptography and Security",
    "cs.DS": "Data Structures and Algorithms",
    "cs.DB": "Databases",
    "cs.SI": "Social and Information Networks",
    "cs.MM": "Multimedia",
    "cs.IT": "Information Theory",
    "cs.PF": "Performance",
    "cs.MA": "Multiagent Systems",
}

# =========================
# Data structures
# =========================

@dataclass
class LLMConfig:
    api_key: str
    model: str
    api_base: Optional[str]
    provider: str = "openai"  # "openai", "gemini", "groq", or "free_local"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    email_domains: List[str]
    abstract: str
    submitted_date: datetime
    pdf_url: str
    arxiv_url: str
    predicted_citations: Optional[float] = None
    prediction_explanations: Optional[List[str]] = None
    semantic_relevance: Optional[float] = None
    semantic_reason: Optional[str] = None
    focus_label: Optional[str] = None
    llm_relevance_score: Optional[float] = None
    venue: Optional[str] = None
    source: Optional[str] = None
    retrieval_source: Optional[str] = None   # "bm25_only" | "faiss_only" | "both" (LanceDB full-text | vector)
    bm25_rank: Optional[int] = None
    faiss_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    has_code: bool = False
    has_dataset: bool = False
    reproducibility_score: int = 0
    paper_type_tag: Optional[str] = None
    cross_encoder_score: Optional[float] = None


# =========================
# Config / secrets
# =========================

def _get_secret(key: str, default: str = "") -> str:
    """Env-only (Cloud Run) — no st.secrets here, backend has no Streamlit runtime."""
    return os.getenv(key, default).strip().strip('\'"')


# =========================
# Corpus access helpers
# =========================

def get_date_range(option: str) -> (date, date):
    today = date.today()
    if option == "Last 3 Days":
        return today - timedelta(days=3), today
    elif option == "Last Week":
        return today - timedelta(days=7), today
    elif option == "Last Month":
        return today - timedelta(days=30), today
    elif option == "All Time":
        return date(2000, 1, 1), today
    else:
        raise ValueError(f"Unknown date range option: {option}")


def get_corpus_dir():
    import pathlib
    import tempfile
    env_dir = os.environ.get("CORPUS_DATA_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    temp_path = pathlib.Path(tempfile.gettempdir()) / "researchagent_corpus"
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


# Process-level state, replacing app.py's st.session_state — a backend process
# is a singleton, so this plays the same "once per running instance" role.
_STATE: Dict[str, Any] = {}


def _check_corpus_freshness(state: Optional[Dict[str, Any]] = None) -> None:
    """Check R2 for corpus updates at most every 30 min per process; reloads in-place."""
    import boto3

    state = _STATE if state is None else state
    now = time.time()
    last_check = state.get("_freshness_checked_at", 0)
    if now - last_check < 1800:
        return

    access_key = _get_secret("R2_ACCESS_KEY_ID")
    secret_key = _get_secret("R2_SECRET_ACCESS_KEY")
    endpoint = _get_secret("R2_ENDPOINT")
    bucket_name = _get_secret("R2_BUCKET")
    if not all([access_key, secret_key, endpoint, bucket_name]):
        return

    try:
        from botocore.config import Config
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        response = s3.head_object(Bucket=bucket_name, Key="corpus/build_meta.json")
        new_etag = response.get("ETag", "").strip('"')
        old_etag = state.get("_corpus_etag")

        if old_etag and new_etag == old_etag:
            state["_freshness_checked_at"] = now
            return

        if old_etag:
            logging.info("[corpus] New corpus data available — refreshing (LanceDB reconnects lazily)...")
            get_lancedb_table.cache_clear()

        state["_corpus_etag"] = new_etag
        state["_freshness_checked_at"] = now
    except Exception:
        # Fail silently: data refresh is non-critical, we don't want to block the search
        pass


def download_corpus_artifacts(state: Optional[Dict[str, Any]] = None) -> None:
    """
    Startup check: download build_meta.json from R2 for freshness tracking.
    LanceDB reads corpus data lazily directly from R2 — no file downloads needed.
    Runs once per process using _STATE as a guard.
    """
    state = _STATE if state is None else state
    if state.get("_corpus_synced"):
        return

    if os.environ.get("CORPUS_DATA_DIR"):
        state["_corpus_synced"] = True
        return

    key_id = _get_secret("R2_ACCESS_KEY_ID")
    access_key = _get_secret("R2_SECRET_ACCESS_KEY")
    endpoint = _get_secret("R2_ENDPOINT")
    bucket = _get_secret("R2_BUCKET")
    if not all([key_id, access_key, endpoint, bucket]):
        state["_corpus_synced"] = True
        return

    corpus_dir = get_corpus_dir()
    corpus_dir.mkdir(parents=True, exist_ok=True)
    meta_path = corpus_dir / "build_meta.json"

    try:
        import boto3
        from botocore.config import Config
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=access_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        s3.download_file(bucket, "corpus/build_meta.json", str(meta_path))
        state["_corpus_synced"] = True

        if not state.get("_models_warmed"):
            try:
                get_specter2_model()
                get_cross_encoder_model()
                get_local_embed_model()
                state["_models_warmed"] = True
                logging.info("[warmup] All models pre-loaded successfully.")
            except Exception as _e:
                logging.warning("[warmup] Model pre-load failed (non-fatal): %s", _e)
    except Exception as e:
        # Do NOT set _corpus_synced — next call retries. LanceDB still connects to
        # R2 directly even without build_meta.json, so this only affects freshness
        # tracking + warm-up timing, not availability.
        logging.warning("[download_corpus_artifacts] build_meta.json sync failed: %s", e)


def build_query_brief(research_brief: str, not_looking_for: str) -> str:
    research_brief = research_brief.strip()
    not_looking_for = not_looking_for.strip()
    parts = []
    if research_brief:
        parts.append("RESEARCH BRIEF:\n" + research_brief)
    if not_looking_for:
        parts.append("WHAT I AM NOT LOOKING FOR:\n" + not_looking_for)
    if not parts:
        return "The user did not provide any research brief."
    return "\n\n".join(parts)


def parse_not_terms(not_text: str) -> List[str]:
    not_text = not_text.strip()
    if not not_text:
        return []
    parts = re.split(r"[,\n;]+", not_text)
    return [p.strip().lower() for p in parts if p.strip()]


def filter_papers_by_not_terms(papers: List[Paper], not_text: str) -> (List[Paper], int):
    terms = parse_not_terms(not_text)
    if not terms or not papers:
        return papers, 0

    filtered: List[Paper] = []
    removed = 0
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        if any(term in text for term in terms):
            removed += 1
        else:
            filtered.append(p)
    return filtered, removed


def filter_papers_by_venue(
    papers: List[Paper],
    venue_filter_type: str,
    selected_category: Optional[str],
    selected_venues: List[str],
):
    if venue_filter_type == "None":
        return papers

    filtered = []
    for p in papers:
        venue = (p.venue or "").lower()
        if venue_filter_type == "All Conferences":
            if any(conf.lower() in venue for conf in CONFERENCE_KEYWORDS):
                filtered.append(p)
        elif venue_filter_type == "All Journals":
            if any(j.lower() in venue for j in JOURNAL_KEYWORDS):
                filtered.append(p)
        elif venue_filter_type == "Specific Venue":
            if selected_category in ("Conference", "Journal"):
                if selected_venues and any(sel.lower() in venue for sel in selected_venues):
                    filtered.append(p)
    return filtered


def extract_venue(comment: str) -> Optional[str]:
    if not comment:
        return None
    c = comment.lower()
    if any(sig in c for sig in NEGATIVE_VENUE_SIGNALS):
        return None
    for venue in sorted(CONFERENCE_KEYWORDS + JOURNAL_KEYWORDS):
        if venue.lower() in c:
            return venue
    return None


def fetch_papers_from_lancedb(
    start_date: date,
    end_date: date,
    category_filter: Optional[str] = None,
    subcats: Optional[List[str]] = None,
) -> List[Paper]:
    """Fetch papers from LanceDB on R2. Most-recent first, capped at DB_FETCH_LIMIT."""
    table = get_lancedb_table()
    if table is None:
        return []

    from data_pipeline.schema import _escape_sql

    filter_parts = [
        f"submitted_date >= '{start_date.isoformat()}'",
        f"submitted_date <= '{end_date.isoformat()}T23:59:59'",
    ]
    if category_filter and category_filter != "All":
        filter_parts.append(f"fields_of_study LIKE '%{_escape_sql(category_filter)}%'")

    if subcats and category_filter != "All":
        or_clauses = []
        for cat_code in subcats:
            cat_name = ARXIV_CODE_TO_NAME.get(cat_code, cat_code)
            words = [w for w in cat_name.split() if w.lower() not in ("and", "or", "of")]
            if words:
                keyword = _escape_sql(words[0] if len(words) == 1 else " ".join(words[:2]))
                safe_code = _escape_sql(cat_code)
                or_clauses.append(
                    f"(title LIKE '%{keyword}%' OR abstract LIKE '%{keyword}%' "
                    f"OR fields_of_study LIKE '%{safe_code}%')"
                )
        if or_clauses:
            filter_parts.append("(" + " OR ".join(or_clauses) + ")")

    where_str = " AND ".join(filter_parts)
    cols = [
        "arxiv_id", "title", "abstract", "authors", "submitted_date",
        "pdf_url", "arxiv_url", "venue", "source", "fields_of_study",
    ]

    try:
        from data_pipeline.schema import query_table
        # ponytail: no server-side ORDER BY/LIMIT push-down on this lance scan (text
        # columns only, no vector) — fine at current corpus size; if "All Time" + "All"
        # category ever OOMs, page via submitted_date instead of scanning-then-head().
        df = query_table(table, filter=where_str, columns=cols)
    except Exception as exc:
        logging.error("[fetch_papers_from_lancedb] Query failed: %s", exc)
        return []

    df = df.sort_values("submitted_date", ascending=False).head(DB_FETCH_LIMIT)

    papers: List[Paper] = []
    for _, r in df.iterrows():
        raw = str(r.get("submitted_date") or "")
        date_str = raw if (len(raw) >= 4 and raw[:4].isdigit()) else "2024-01-01"
        if "T" not in date_str:
            date_str = date_str + "T00:00:00"
        try:
            submitted = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            submitted = datetime(2024, 1, 1)
        papers.append(Paper(
            arxiv_id=str(r.get("arxiv_id") or ""),
            title=str(r.get("title") or ""),
            authors=json.loads(str(r.get("authors") or "[]")),
            email_domains=[],
            abstract=str(r.get("abstract") or ""),
            submitted_date=submitted,
            pdf_url=str(r.get("pdf_url") or ""),
            arxiv_url=str(r.get("arxiv_url") or ""),
            venue=r.get("venue") or None,
            source=r.get("source") or None,
        ))
    return papers


# =========================
# Generic LLM call + JSON helper
# =========================

def call_llm(prompt: str, llm_config: LLMConfig, label: str = "") -> str:
    if not llm_config or not llm_config.api_key or not llm_config.api_key.strip():
        return ""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if llm_config.provider == "openai":
                from openai import OpenAI
                client_args = {"api_key": llm_config.api_key}
                if llm_config.api_base and llm_config.api_base.strip():
                    client_args["base_url"] = llm_config.api_base
                client = OpenAI(**client_args)

                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ]
                kwargs: Dict[str, Any] = {"model": llm_config.model, "messages": messages}
                if not (llm_config.model.startswith("o1") or llm_config.model.startswith("gpt-5")):
                    kwargs["temperature"] = 0.2
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content

            elif llm_config.provider == "gemini":
                if genai is None:
                    raise RuntimeError("Gemini provider selected but google-genai package is not installed.")
                client = genai.Client(api_key=llm_config.api_key)
                response = client.models.generate_content(model=llm_config.model, contents=prompt)
                if hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                        texts = [part.text for part in cand.content.parts if getattr(part, "text", None)]
                        if texts:
                            return "".join(texts)
                return getattr(response, "text", "")

            elif llm_config.provider == "groq":
                from groq import Groq
                client = Groq(api_key=llm_config.api_key)
                response = client.chat.completions.create(
                    model=llm_config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content

            else:
                raise ValueError(f"Unknown provider: {llm_config.provider}")

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM call failed ({label}): {e}")
                raise e
            time.sleep(2 * (attempt + 1))


def safe_parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        return None
    return None


# =========================
# Embeddings
# =========================

@lru_cache(maxsize=1)
def get_local_embed_model():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_texts_local(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_local_embed_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1)
    norm2 = sum(b * b for b in vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


@lru_cache(maxsize=1)
def get_lancedb_table():
    """Connect to LanceDB (R2 via env vars, or local CORPUS_DATA_DIR) and return the papers table."""
    local_path = os.environ.get("CORPUS_DATA_DIR") or None
    try:
        from data_pipeline.schema import connect_lancedb, get_or_create_papers_table
        db = connect_lancedb(local_path)
        return get_or_create_papers_table(db)
    except Exception as exc:
        logging.error("[LanceDB] Connection failed: %s", exc)
        return None


# =========================
# Task 34 — SPECTER2 adhoc_query Stage 2 adapter
# =========================

@lru_cache(maxsize=1)
def get_specter2_model():
    """
    Load SPECTER2 base + adhoc_query adapter for asymmetric scientific retrieval.
    Returns (model, tokenizer) on success, (None, None) on any failure so the
    caller can fall back to the MiniLM pre-computed embedding path.
    """
    try:
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base", torch_dtype=torch.float16)

        model.load_adapter(
            "allenai/specter2_adhoc_query",
            source="hf",
            load_as="specter2_adhoc_query",
            set_active=True,
        )

        # load_adapter() adds adapter weights in float32 regardless of the base
        # model's dtype — recast the whole model (base + adapter) so forward
        # passes don't hit a Half/Float matmul mismatch. .to() can reset the
        # adapter routing state, so (re)activate AFTER this, not before.
        model = model.to(torch.float16)
        model.set_active_adapters("specter2_adhoc_query")

        try:
            active_adapters = model.active_adapters
            if "specter2_adhoc_query" not in active_adapters:
                print(f"[SPECTER2] Warning: Adapter not active. Active adapters: {active_adapters}")
                model.set_active_adapters("specter2_adhoc_query")
        except Exception as e:
            print(f"[SPECTER2] Warning: Could not verify adapter activation: {e}")

        model.eval()
        return model, tokenizer
    except ImportError:
        print("[SPECTER2] 'adapters' library not installed — falling back to MiniLM Stage 2.")
        return None, None
    except Exception as e:
        print(f"[SPECTER2] load error: {e} — falling back to MiniLM Stage 2.")
        return None, None


def specter2_vector_rerank(
    papers: List[Paper],
    query_brief: str,
    n2: int = 300,
    paper_vectors: Optional[Dict[str, list]] = None,
) -> List[Paper]:
    """
    Stage 2 reranker — asymmetric SPECTER2 retrieval with a LanceDB-vector fast path.
    Fast path:   paper_vectors provided -> ~0.05s for 1200 papers
    Slow path:   paper_vectors empty -> live SPECTER2 encode for all papers (~60s/1200)
    Fallback:    SPECTER2 model unavailable -> return [] so caller uses MiniLM path
    Sets p.semantic_relevance on each paper.
    """
    if not papers:
        return []

    model, tokenizer = get_specter2_model()
    if model is None or tokenizer is None:
        return []

    import torch

    try:
        model.set_active_adapters("specter2_adhoc_query")
    except Exception as e:
        print(f"[SPECTER2] Warning: Could not set active adapter: {e}")

    def _encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt",
                max_length=512, return_token_type_ids=False,
            )
            with torch.no_grad():
                output = model(**inputs)
            vecs = output.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    try:
        q_vecs = _encode_batch([query_brief])
        q_vec = q_vecs[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        have_lancedb_vecs = bool(paper_vectors)

        if have_lancedb_vecs:
            indexed_papers: List[Paper] = []
            indexed_vecs_list: List[np.ndarray] = []
            unindexed_papers: List[Paper] = []

            for p in papers:
                vec = (paper_vectors or {}).get(p.arxiv_id)
                if vec is not None:
                    arr = np.array(vec, dtype="float32")
                    if arr.shape[0] == q_vec.shape[0]:
                        indexed_papers.append(p)
                        indexed_vecs_list.append(arr)
                        continue
                unindexed_papers.append(p)

            if indexed_papers:
                indexed_vecs = np.vstack(indexed_vecs_list).astype("float32")
                norms = np.linalg.norm(indexed_vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                indexed_vecs = indexed_vecs / norms
                indexed_sims = indexed_vecs @ q_vec
            else:
                indexed_sims = np.array([], dtype="float32")

            if unindexed_papers:
                sep = tokenizer.sep_token or " "
                unindexed_texts = [p.title + sep + (p.abstract if p.abstract else "") for p in unindexed_papers]
                unindexed_vecs = _encode_batch(unindexed_texts)
                norms = np.linalg.norm(unindexed_vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                unindexed_vecs = unindexed_vecs / norms
                unindexed_sims = unindexed_vecs @ q_vec
            else:
                unindexed_sims = np.array([], dtype="float32")

            all_papers = indexed_papers + unindexed_papers
            all_sims = np.concatenate([indexed_sims, unindexed_sims])
            scored = []
            for p, sim in zip(all_papers, all_sims):
                p.semantic_relevance = float(sim)
                scored.append((float(sim), p))

            n_live = len(unindexed_papers)
            if n_live:
                print(f"[SPECTER2 Stage 2] fast-path: {len(indexed_papers)} precomputed, {n_live} live-encoded")
        else:
            sep = tokenizer.sep_token or " "
            paper_texts = [p.title + sep + (p.abstract if p.abstract else "") for p in papers]
            p_vecs = _encode_batch(paper_texts)
            norms = np.linalg.norm(p_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            p_vecs = p_vecs / norms
            all_sims = p_vecs @ q_vec
            scored = []
            for p, sim in zip(papers, all_sims):
                p.semantic_relevance = float(sim)
                scored.append((float(sim), p))

        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(n2, len(scored))
        return [p for _, p in scored[:k]]
    except Exception as e:
        print(f"[SPECTER2] rerank inference error: {e} — caller will use MiniLM fallback.")
        return []


def minilm_vector_rerank(papers: List[Paper], query_brief: str, n2: int = 300) -> List[Paper]:
    """Stage 2 MiniLM fallback — runtime-embeds papers + query when SPECTER2 is unavailable."""
    if not papers:
        return []
    texts = [p.title + "\n\n" + p.abstract for p in papers]
    paper_vecs = embed_texts_local(texts)
    q_vec = embed_texts_local([query_brief])[0]
    scored = []
    for p, vec in zip(papers, paper_vecs):
        sim = cosine_similarity(q_vec, vec)
        p.semantic_relevance = sim
        scored.append((sim, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    k = min(n2, len(scored))
    return [p for _, p in scored[:k]]


def _lancedb_hybrid_stage1(
    table,
    papers: List[Paper],
    fts_query: str,
    q_vec: Optional[np.ndarray],
    top_k: int = 400,
    quality_where: Optional[str] = None,
) -> tuple:
    """
    LanceDB FTS + vector search with manual RRF.
    Returns (stage1_papers, paper_vectors). paper_vectors: {arxiv_id: list[float]}.
    Search scoped globally then intersected with the `papers` pool (already date-filtered).
    """
    if not papers:
        return [], {}

    paper_ids = {p.arxiv_id for p in papers}
    paper_dict = {p.arxiv_id: p for p in papers}
    fts_ranks: Dict[str, int] = {}
    vec_ranks: Dict[str, int] = {}
    paper_vectors: Dict[str, list] = {}

    search_k = min(top_k * 5, 6000)

    pool_date_where: Optional[str] = None
    if papers:
        dates = [p.submitted_date.strftime("%Y-%m-%d") for p in papers if p.submitted_date]
        if dates:
            min_date, max_date = min(dates), max(dates)
            pool_date_where = f"submitted_date >= '{min_date}' AND submitted_date <= '{max_date}T23:59:59'"

    scope_where_parts = [p for p in (pool_date_where, quality_where) if p]
    scope_where = " AND ".join(f"({p})" for p in scope_where_parts) if scope_where_parts else None

    try:
        fts_q = (
            table.search(fts_query, query_type="fts")
            .select(["arxiv_id", "has_embedding", "vector"])
            .limit(search_k)
        )
        if scope_where:
            fts_q = fts_q.where(scope_where)
        fts_rows = fts_q.to_list()
        rank = 1
        for row in fts_rows:
            aid = row.get("arxiv_id")
            if aid and aid in paper_ids and aid not in fts_ranks:
                fts_ranks[aid] = rank
                if row.get("has_embedding") and row.get("vector") is not None:
                    paper_vectors[aid] = row["vector"]
                rank += 1
    except Exception as exc:
        logging.warning("[Stage1/FTS] LanceDB FTS failed: %s", exc)

    if q_vec is not None:
        try:
            vec_where = "has_embedding = true"
            if scope_where:
                vec_where = f"has_embedding = true AND ({scope_where})"
            vec_rows = (
                table.search(q_vec.tolist(), query_type="vector")
                .where(vec_where, prefilter=True)
                .select(["arxiv_id", "vector"])
                .limit(search_k)
                .to_list()
            )
            rank = 1
            for row in vec_rows:
                aid = row.get("arxiv_id")
                if aid and aid in paper_ids and aid not in vec_ranks:
                    vec_ranks[aid] = rank
                    if aid not in paper_vectors and row.get("vector") is not None:
                        paper_vectors[aid] = row["vector"]
                    rank += 1
        except Exception as exc:
            logging.warning("[Stage1/Vec] LanceDB vector search failed: %s", exc)

    union_ids = set(fts_ranks.keys()) | set(vec_ranks.keys())
    n_fts, n_vec = len(fts_ranks), len(vec_ranks)

    scored: List[tuple] = []
    for aid in union_ids:
        rrf = 1.0 / (RRF_K + fts_ranks.get(aid, n_fts + 1)) + 1.0 / (RRF_K + vec_ranks.get(aid, n_vec + 1))
        scored.append((rrf, aid))
    scored.sort(reverse=True)

    result_papers: List[Paper] = []
    for rrf_score, aid in scored[:top_k]:
        p = paper_dict.get(aid)
        if p:
            p.bm25_rank = fts_ranks.get(aid)
            p.faiss_rank = vec_ranks.get(aid)
            p.rrf_score = rrf_score
            p.retrieval_source = (
                "both" if aid in fts_ranks and aid in vec_ranks
                else "bm25_only" if aid in fts_ranks
                else "faiss_only"
            )
            result_papers.append(p)

    return result_papers, paper_vectors


@lru_cache(maxsize=1)
def get_cross_encoder_model():
    try:
        from sentence_transformers import CrossEncoder
        import torch
        # A2 (docs/relevance-strategy-comparison.md): swapped from bge-reranker-base
        # to bge-reranker-v2-m3 - same family, real BEIR nDCG@10 ~56.4 vs ~49.5,
        # and a higher recommended max_length (1024 vs 512) so less of each
        # abstract gets truncated. model_kwargs replaces the deprecated automodel_args.
        logging.info("[CrossEncoder] Loading BAAI/bge-reranker-v2-m3 for precision re-ranking (first run only)...")
        return CrossEncoder("BAAI/bge-reranker-v2-m3", model_kwargs={"torch_dtype": torch.float16})
    except Exception as e:
        print(f"CrossEncoder load error: {e}")
        return None


def cross_encoder_rerank(papers: List[Paper], query_brief: str, n3: int = 150) -> List[Paper]:
    """
    Writes sigmoid(logit) exclusively to p.cross_encoder_score. Does NOT overwrite
    p.semantic_relevance (Stage 2 cosine — preserved for display + fallback).
    """
    if not papers:
        return []
    model = get_cross_encoder_model()
    if not model:
        return papers[:n3]

    pairs = [
        [query_brief, p.title + "\n\n" + (p.abstract or "")[:_CE_MAX_ABSTRACT_CHARS]]
        for p in papers
    ]
    try:
        scores = model.predict(pairs)
        scored = []
        for p, score in zip(papers, scores):
            score_float = float(score)
            p.cross_encoder_score = 1 / (1 + math.exp(-score_float))
            scored.append((score_float, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(n3, len(scored))
        return [p for _, p in scored[:k]]
    except Exception as e:
        print(f"CrossEncoder predict error: {e}")
        return papers[:n3]


def _default_emit(msg: str, level: str = "write") -> None:
    logging.info("[pipeline/%s] %s", level, msg)


def select_embedding_candidates(
    papers: List[Paper],
    query_brief: str,
    llm_config: Optional[LLMConfig] = None,
    max_candidates: int = 150,
    emit: Callable[[str, str], None] = _default_emit,
    qil_cache: Optional[dict] = None,
) -> List[Paper]:
    """
    4-stage hybrid search pipeline:
      Stage 0 — QIL decomposes brief -> semantic_query, bm25_keywords, intent
      Stage 1 — LanceDB full-text + vector search -> RRF -> adaptive top-K
      Stage 2 — SPECTER2 vector rerank -> top-400 (MiniLM fallback)
      Stage 3 — CrossEncoder precision rerank -> top-150
    Graceful degradation: full-text-only, vector-only, or full-pool fallback.

    `emit(msg, level)` defaults to a logging-based no-op UI; pass a callback
    (or app.py's `_st_emit`) to route progress messages elsewhere. `qil_cache`
    defaults to a fresh per-call dict; pass a persistent dict to keep QIL
    cross-call caching (this is the exact seam introduced for the 2026-07-16
    architecture deepening — see tests/test_select_embedding_candidates.py).
    """
    if not papers:
        return []
    if qil_cache is None:
        qil_cache = {}

    emit(f"Starting 3-stage hybrid search from {len(papers)} LanceDB candidates...")

    # ─── Stage 0: Query Intelligence Layer ────────────────────────────────
    sq = None
    if analyse_query is not None:
        try:
            _qil_cache_key = hashlib.md5(query_brief.strip().lower().encode()).hexdigest()
            sq = qil_cache.get(_qil_cache_key)

            if sq is None:
                groq_key = os.getenv("GROQ_API_KEY", "").strip()
                if not groq_key and llm_config and llm_config.provider == "groq" and llm_config.api_key:
                    groq_key = llm_config.api_key.strip()
                or_key = _get_secret("OPENROUTER_API_KEY")

                sq = analyse_query(
                    brief=query_brief,
                    groq_api_key=groq_key or None,
                    openrouter_api_key=or_key or None,
                )
                qil_cache[_qil_cache_key] = sq

            _source_label = {
                "llm_groq": "LLM/Groq",
                "llm_openrouter": "LLM/OpenRouter",
                "rules": "Rules",
            }.get(sq.source, sq.source.upper())
            emit(
                f"⏱ Stage 0 (QIL/{_source_label}): intent=`{sq.intent}` · "
                f"quality=`{sq.quality_modifier}` · "
                f"keywords=`{', '.join(sq.bm25_keywords[:4])}{'...' if len(sq.bm25_keywords) > 4 else ''}`"
            )
        except Exception as _qil_err:
            print(f"[QIL] query analysis failed: {_qil_err} — using raw brief")
            sq = None

    bm25_query = sq.bm25_query_string if sq and sq.bm25_keywords else None
    semantic_query = sq.semantic_query if sq and sq.semantic_query else None

    # ─── Stage 1: LanceDB FTS + vector search → manual RRF ───────────────
    adaptive_k = min(int(len(papers) * 0.07), 1200)
    adaptive_k = max(adaptive_k, 50)
    emit(f"⏱ Stage 1: LanceDB FTS + vector → RRF merge (adaptive_k={adaptive_k})...")

    q_vec_stage1: Optional[np.ndarray] = None
    try:
        import torch
        specter2_model_s1, specter2_tok_s1 = get_specter2_model()
        if specter2_model_s1 is not None and specter2_tok_s1 is not None:
            encode_text = semantic_query if semantic_query and semantic_query.strip() else query_brief
            inputs = specter2_tok_s1(
                [encode_text], padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                out = specter2_model_s1(**inputs)
            q_vec_stage1 = out.last_hidden_state[:, 0, :].cpu().float().numpy()[0]
            norm = np.linalg.norm(q_vec_stage1)
            if norm > 0:
                q_vec_stage1 = q_vec_stage1 / norm
    except Exception as _qv_err:
        logging.warning("[Stage1] query vector generation failed: %s", _qv_err)
        q_vec_stage1 = None

    fts_query_str = bm25_query if bm25_query else query_brief
    lancedb_table = get_lancedb_table()
    paper_vectors: Dict[str, list] = {}

    quality_where = QUALITY_LANCEDB_FILTERS.get(sq.quality_modifier) if sq else None
    if quality_where:
        emit(f"⏱ Stage 1: quality filter active — `{quality_where}`")

    if lancedb_table is not None:
        stage1_papers, paper_vectors = _lancedb_hybrid_stage1(
            lancedb_table, papers, fts_query_str, q_vec_stage1, top_k=adaptive_k,
            quality_where=quality_where,
        )
        if len(stage1_papers) < 50:
            emit(
                f"Stage 1 returned {len(stage1_papers)} candidates (below 50 threshold) "
                "— retrying with a wider search before falling back.",
                "info",
            )
            # Reuses _lancedb_hybrid_stage1's own search_k cap (min(top_k*5, 6000));
            # top_k=1200 pushes it to the max, so this is just a wider single query,
            # not a new query mechanism.
            stage1_papers, paper_vectors = _lancedb_hybrid_stage1(
                lancedb_table, papers, fts_query_str, q_vec_stage1, top_k=1200,
                quality_where=quality_where,
            )
            if len(stage1_papers) < 50:
                emit(
                    f"Stage 1 still only {len(stage1_papers)} candidates after widening "
                    "— using full pool as fallback.",
                    "info",
                )
                stage1_papers = papers
                paper_vectors = {}
        else:
            n_both = sum(1 for p in stage1_papers if p.retrieval_source == "both")
            n_fts_only = sum(1 for p in stage1_papers if p.retrieval_source == "bm25_only")
            n_vec_only = sum(1 for p in stage1_papers if p.retrieval_source == "faiss_only")
            emit(
                f"✅ RRF Stage 1: {len(stage1_papers)} candidates — "
                f"🔵 {n_both} both · 🟠 {n_fts_only} FTS-only · 🟣 {n_vec_only} vector-only"
            )
    else:
        emit("⚠️ LanceDB unavailable — using full pool as Stage 1 fallback.", "warning")
        stage1_papers = papers

    # ─── P-10: QIL not_terms filter — applied once after Stage 1 ──────────
    qil_not_terms = (sq.hard_filters.get("not_terms", []) if sq else [])
    if qil_not_terms:
        before_not = len(stage1_papers)
        stage1_papers = [
            p for p in stage1_papers
            if not any(t in (p.title + " " + (p.abstract or "")).lower() for t in qil_not_terms)
        ]
        removed_not = before_not - len(stage1_papers)
        if removed_not:
            emit(f"🚫 QIL not_terms filter removed {removed_not} papers from Stage 1 pool.")

    # ─── Stage 2: SPECTER2 adhoc_query → MiniLM fallback ─────────────────
    stage2_query = semantic_query if semantic_query else query_brief
    have_stage1_vecs = bool(paper_vectors)
    emit(
        f"⏱ Stage 2: SPECTER2 semantic reranking "
        f"({'LanceDB vector fast-path' if have_stage1_vecs else 'live-encode'})..."
    )
    stage2_papers = specter2_vector_rerank(stage1_papers, stage2_query, n2=200, paper_vectors=paper_vectors)
    if stage2_papers:
        emit(f"✅ SPECTER2 Stage 2: {len(stage2_papers)} candidates selected (scientific asymmetric retrieval).")
    else:
        emit(
            "⚠️ SPECTER2 unavailable — falling back to MiniLM for Stage 2. "
            "Retrieval quality may be reduced. Check that `allenai/specter2` is installed.",
            "warning",
        )
        stage2_papers = minilm_vector_rerank(stage1_papers, stage2_query, n2=200)
        emit(f"✅ MiniLM Stage 2 fallback: {len(stage2_papers)} candidates selected.")

    # ─── Stage 3: CrossEncoder Precision Rerank ───────────────────────
    stage3_query = semantic_query if semantic_query else query_brief
    emit("⏱ Stage 3: Cross-Encoder precision reranking...")
    stage3_papers = cross_encoder_rerank(stage2_papers, stage3_query, n3=max_candidates)
    emit(f"✅ Cross-Encoder selected {len(stage3_papers)} final candidates.")

    # ─── Stage 4: Abstract Highlights Extraction ──────────────────────
    emit("⏱ Stage 4: Extracting sentence-level abstract highlights...")
    stage3_papers = extract_abstract_highlights(stage3_papers, stage2_query)

    # ─── Stage 5: Artifact signal detection (Task 38) ─────────────────
    stage3_papers = enrich_paper_signals(stage3_papers)

    return stage3_papers


# =========================
# Task 36 — Stage 4 Abstract Highlights
# =========================

def extract_abstract_highlights(papers: List[Paper], query_brief: str) -> List[Paper]:
    """
    Splits paper abstracts into sentences, scores them against the query_brief
    using the lightweight local MiniLM embedding model, and sets p.semantic_reason
    to the top-2 most relevant sentences.
    """
    if not papers:
        return papers

    try:
        model = get_local_embed_model()
        q_vec = model.encode([query_brief], normalize_embeddings=True)[0]
    except Exception as e:
        print(f"Failed to load embed model for abstract highlights: {e}")
        return papers

    sent_regex = re.compile(r"(?<=[.!?])\s+")

    for p in papers:
        if not p.abstract:
            continue
        sentences = [s.strip() for s in sent_regex.split(p.abstract) if len(s.strip()) >= 15]
        if not sentences:
            p.semantic_reason = "No extractable sentences found in abstract."
            continue
        try:
            s_vecs = model.encode(sentences, normalize_embeddings=True)
            scores = [float(np.dot(vec, q_vec)) for vec in s_vecs]
            scored_sentences = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
            top_sents = [s for _, s in scored_sentences[:2]]
            if len(top_sents) == 1:
                p.semantic_reason = f"Matched: '{top_sents[0][:120]}...'"
            else:
                p.semantic_reason = f"Matched: '{top_sents[0][:120]}...' | '{top_sents[1][:120]}...'"
        except Exception as e:
            print(f"Highlight extraction error for {p.arxiv_id}: {e}")

    return papers


# =========================
# Task 38 — Artifact signal detection and paper type tagging
# =========================

def enrich_paper_signals(papers: List[Paper]) -> List[Paper]:
    """Sets artifact signals (has_code, has_dataset, reproducibility_score, paper_type_tag)."""
    for p in papers:
        abstract_lower = p.abstract.lower() if p.abstract else ""
        pdf_lower = p.pdf_url.lower() if p.pdf_url else ""

        if "github.com/" in abstract_lower or "github.com/" in pdf_lower:
            p.has_code = True

        dataset_keywords = ["new dataset", "we release", "we introduce a benchmark", "data collection", "we collected"]
        if any(kw in abstract_lower for kw in dataset_keywords):
            p.has_dataset = True

        score = 0
        if "ablation" in abstract_lower:
            score += 1
        if "code available" in abstract_lower or "github.com/" in abstract_lower:
            score += 1
        if "reproducib" in abstract_lower:
            score += 1
        p.reproducibility_score = score

        if any(kw in abstract_lower for kw in ["survey", "comprehensive study", "we survey"]):
            p.paper_type_tag = "Survey"
        elif any(kw in abstract_lower for kw in ["new dataset", "we introduce a benchmark", "we collect"]):
            p.paper_type_tag = "New Dataset"
        elif any(kw in abstract_lower for kw in ["we scale", "billion parameter", "x larger", "× larger"]):
            p.paper_type_tag = "Scale Study"
        elif any(kw in abstract_lower for kw in ["we propose", "novel method", "new architecture", "new approach"]):
            p.paper_type_tag = "New Technique"
        elif any(kw in abstract_lower for kw in ["we evaluate", "empirical study", "analysis of"]):
            p.paper_type_tag = "Evaluation"
        else:
            p.paper_type_tag = "Other"

    return papers


# =========================
# SciBERT Split Classification (Tasks 41 & 42)
# =========================

def scibert_classify_papers(papers: List[Paper]) -> List[Paper]:
    """
    Primary path (CE available): reads p.cross_encoder_score, thresholds at
    PRIMARY_THRESHOLD=0.55 / SECONDARY_THRESHOLD=0.25.
    Fallback path (CE unavailable): delegates to heuristic_classify_papers_free()
    (rank-based top-30% = primary over p.semantic_relevance).
    """
    ce_available = any(p.cross_encoder_score is not None for p in papers)
    if not ce_available:
        return heuristic_classify_papers_free(papers)

    for p in papers:
        score = p.cross_encoder_score if p.cross_encoder_score is not None else 0.0
        p.llm_relevance_score = score
        if score >= PRIMARY_THRESHOLD:
            p.focus_label = "primary"
        elif score >= SECONDARY_THRESHOLD:
            p.focus_label = "secondary"
        else:
            p.focus_label = "off-topic"
    return papers


def heuristic_classify_papers_free(candidates: List[Paper]) -> List[Paper]:
    if not candidates:
        return candidates
    ranked = sorted(candidates, key=lambda p: p.semantic_relevance or 0.0, reverse=True)
    n = len(ranked)
    if n == 0:
        return ranked
    top_k = max(1, min(n, max(10, int(0.3 * n))))
    for idx, p in enumerate(ranked):
        p.llm_relevance_score = p.semantic_relevance or 0.0
        p.focus_label = "primary" if idx < top_k else "secondary"
        if p.semantic_reason is None:
            p.semantic_reason = "Heuristic classification based on embedding similarity."
    return ranked


# =========================
# MONEYBALL Impact Scoring
# =========================

def get_s2_citation_stats(paper: Paper, api_key: Optional[str] = None) -> int:
    """Return max author citation count from Semantic Scholar; 0 if unavailable."""
    headers = {"x-api-key": api_key} if api_key else {}
    max_retries = 2

    def fetch(url, params):
        for attempt in range(max_retries + 1):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=10)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(2 * (attempt + 1))
            except requests.RequestException:
                if attempt < max_retries:
                    time.sleep(1)
        return None

    if paper.arxiv_id:
        data = fetch(
            f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{paper.arxiv_id.split('v')[0]}",
            {"fields": "authors.citationCount"},
        )
        if data:
            auth_cites = [a.get("citationCount", 0) for a in data.get("authors", []) if a.get("citationCount")]
            if auth_cites:
                return max(auth_cites)

    data = fetch(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": paper.title, "limit": 1, "fields": "title,citationCount,authors.citationCount"},
    )
    if data and data.get("data"):
        auth_cites = [a.get("citationCount", 0) for a in data["data"][0].get("authors", []) if a.get("citationCount")]
        return max(auth_cites) if auth_cites else 0
    return 0


def resolve_moneyball_weights(quality_modifier: str = "any") -> dict:
    """
    QIL quality_modifier weight override (docs/qil-improvements-planner.md B1, Job 2)
    takes precedence when set (influential/emerging/classic); "any" and "recent" fall
    through to moneyball_weights.json (trained/calibrated) if present, else the static
    DEFAULT_MONEYBALL_WEIGHTS.
    """
    weights = QUALITY_MONEYBALL_WEIGHTS.get(quality_modifier)
    if weights is not None:
        return weights
    if os.path.exists("moneyball_weights.json"):
        try:
            with open("moneyball_weights.json", "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_MONEYBALL_WEIGHTS


def predict_citations_direct(
    target_papers: List[Paper],
    llm_config: LLMConfig,
    batch_size: int = 8,
    quality_modifier: str = "any",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[Paper]:
    """MONEYBALL PREDICTOR: Hybrid Author Data + Custom LLM Narrative."""
    if not target_papers:
        return target_papers

    weights = resolve_moneyball_weights(quality_modifier)
    s2_key = _get_secret("S2_API_KEY")

    for i, p in enumerate(target_papers):
        max_auth_cites = get_s2_citation_stats(p, s2_key)

        is_fresh = False
        try:
            days_old = (datetime.now().date() - p.submitted_date.date()).days
            if days_old <= 5:
                is_fresh = True
        except Exception:
            pass

        if max_auth_cites > 0:
            h1_fame = min(math.log(max_auth_cites + 1) * 8, 95)
            fame_label = "real"
        elif is_fresh:
            fame_label = "too_new"
            h1_fame = 0.0
        else:
            h1_fame = 0.0
            fame_label = "none"

        if not s2_key:
            time.sleep(0.3)

        t_lower = p.title.lower()
        h2_hype = 0
        if "benchmark" in t_lower or "dataset" in t_lower:
            h2_hype += 50
        if "survey" in t_lower:
            h2_hype += 40
        if "llm" in t_lower:
            h2_hype += 10

        h3_sniper = 0
        if "benchmark" in t_lower:
            h3_sniper += 50
        niche = ["lidar", "3d", "audio", "wireless", "agriculture", "traffic", "physics"]
        if any(n in t_lower for n in niche):
            h3_sniper -= 20

        prompt = textwrap.dedent(f"""
            Analyze this abstract.
            1. Rate 'Citation Potential' (0-10) based on market fit (Broad/Hot = High, Niche = Low).
            2. Write 2 short, plain English sentences explaining the score.
               - Sentence 1 (Market Fit): Why is this topic hot or niche? (Do NOT start with "Market Fit:")
               - Sentence 2 (Contribution): What is the specific value? (Do NOT start with "Contribution:")

            Title: {p.title}
            Abstract: {p.abstract[:800]}...

            Return JSON: {{ "score": <int>, "bullets": ["string", "string"] }}
        """)

        h4_utility = 50.0
        content_bullets = [
            "The topic appears relevant to current research trends.",
            "The paper proposes a specific contribution to the field.",
        ]

        try:
            raw = ""
            if llm_config and llm_config.api_key and llm_config.api_key.strip():
                raw = call_llm(prompt, llm_config, label="moneyball_narrative")

            if raw:
                if "```" in raw:
                    parts = raw.split("```json")
                    raw = parts[1].split("```")[0] if len(parts) > 1 else raw.split("```")[1].split("```")[0]

                parsed = json.loads(raw.strip())
                h4_utility = float(parsed.get("score", 5) * 10)

                if "bullets" in parsed and isinstance(parsed["bullets"], list):
                    content_bullets = [
                        b.replace("Market Fit:", "").replace("Contribution:", "").strip()
                        for b in parsed["bullets"][:2]
                    ]
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        if fame_label == "too_new":
            p.predicted_citations = -1.0  # sentinel: "unrated"
        else:
            score = (
                h1_fame * weights["weight_fame"]
                + h2_hype * weights["weight_hype"]
                + h3_sniper * weights["weight_sniper"]
                + h4_utility * weights["weight_utility"]
            )
            if quality_modifier == "recent":
                score += compute_recency_score(p.submitted_date) * RECENCY_BONUS_MAX
            p.predicted_citations = score

        final_bullets = []
        if fame_label == "real":
            if max_auth_cites > 3000:
                final_bullets.append("🚀 **Distribution:** High influence author/lab.")
            elif max_auth_cites > 500:
                final_bullets.append("📢 **Reach:** Established track record.")
            elif max_auth_cites > 100:
                final_bullets.append("📈 **Momentum:** Authors have prior traction.")
            else:
                final_bullets.append("🌱 **Emerging:** Newer authors; relies on merit.")
        elif fame_label == "too_new":
            final_bullets.append("🆕 **Too new for impact score:** Citation data unavailable. Ranked by relevance only.")
            if p.semantic_reason:
                final_bullets.append(f"✨ **Relevance Insight:** Ranked highly because: {p.semantic_reason}")
        else:
            final_bullets.append("🌱 **Emerging:** Unknown authors.")

        if len(content_bullets) >= 1:
            final_bullets.append(f"🎯 **Market Fit:** {content_bullets[0]}")
        if len(content_bullets) >= 2:
            final_bullets.append(f"💡 **Contribution:** {content_bullets[1]}")

        p.prediction_explanations = final_bullets

        if on_progress:
            on_progress(i + 1, len(target_papers))

    return target_papers


def compute_recency_score(submitted_date: datetime, max_days: int = 30) -> float:
    """Returns a recency score in [0, 1], where newer papers score higher."""
    try:
        days_old = max((datetime.now().date() - submitted_date.date()).days, 0)
    except Exception:
        return 0.0
    if days_old >= max_days:
        return 0.0
    return 1.0 - (days_old / max_days)


def assign_heuristic_citations_free(papers: List[Paper]) -> List[Paper]:
    if not papers:
        return papers
    scores = [(p.llm_relevance_score or 0.0) * 0.7 + (p.semantic_relevance or 0.0) * 0.3 for p in papers]
    if not scores:
        return papers
    min_s, max_s = min(scores), max(scores)
    for p, s in zip(papers, scores):
        norm = (s - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        p.predicted_citations = float(int(10 + norm * 40))
    return papers


def summarize_paper_plain_english(paper: Paper, llm_config: LLMConfig) -> str:
    if not llm_config or not llm_config.api_key or not llm_config.api_key.strip():
        return "Plain English summary not available: Missing API key or configuration."

    prompt = textwrap.dedent(f"""
    Explain this research paper to a non-expert.
    Title: {paper.title}
    Abstract: {paper.abstract}

    Provide 3-6 plain English bullet points covering main idea, problem solved, and takeaways.
    """).strip()
    return call_llm(prompt, llm_config, label="plain_english_summary")
