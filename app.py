import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import hashlib
import json
import logging
import time
import textwrap

# Hugging Face optimization settings (must be set before importing transformers)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # prevents fork warning
_MODEL_DTYPE = "float16"

import threading
_PIPELINE_SEMAPHORE = threading.Semaphore(2)

import io
import zipfile
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Callable
import numpy as np
import pathlib
import tempfile

try:
    import streamlit as st
    import requests
    import feedparser
    import pandas as pd
    from openai import OpenAI, NotFoundError, BadRequestError
    from groq import Groq
except ImportError as e:
    missing = str(e).split("'")[1]
    print(f"Missing package: {missing}")
    print("Please run: pip install streamlit requests feedparser openai pandas groq")
    raise

# Optional local embedding model for free mode
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore

# Optional Google Gemini client
try:
    from google import genai  # type: ignore
    from google.genai import types # type: ignore
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

# MONEYBALL DEFAULTS
DEFAULT_MONEYBALL_WEIGHTS = {
    "weight_fame": 0.84,
    "weight_hype": 0.0,
    "weight_sniper": 0.0,
    "weight_utility": 0.16
}

# QIL quality_modifier -> LanceDB pre-filter (docs/qil-improvements-planner.md B1, Job 1).
# Fires before vector/FTS search narrows the pool. "any" applies no filter.
QUALITY_LANCEDB_FILTERS = {
    "recent":      "year >= 2023",
    "influential": "citation_count >= 50",
    "emerging":    "year >= 2023 AND citation_count < 20",
    "classic":     "year <= 2018",
}

# QIL quality_modifier -> Moneyball weight override (B1, Job 2). Fires after retrieval,
# replaces DEFAULT_MONEYBALL_WEIGHTS/moneyball_weights.json for this scoring pass.
# "recent" has no fixed weight set — it applies an additive recency bonus instead
# (see RECENCY_BONUS_MAX in predict_citations_direct). "any" leaves weights unchanged.
QUALITY_MONEYBALL_WEIGHTS = {
    "influential": {"weight_fame": 0.95, "weight_hype": 0.0, "weight_sniper": 0.0, "weight_utility": 0.05},
    "emerging":    {"weight_fame": 0.30, "weight_hype": 0.0, "weight_sniper": 0.0, "weight_utility": 0.70},
    "classic":     {"weight_fame": 0.90, "weight_hype": 0.0, "weight_sniper": 0.0, "weight_utility": 0.10},
}

# =========================
# SciBERT Split — Classification thresholds (Task 41)
# Applied to cross_encoder_score (dedicated Stage 3 field, sigmoid of BGE-reranker-base logit).
# When CrossEncoder is unavailable, scibert_classify_papers falls back to
# heuristic_classify_papers_free (rank-based top-30%) using semantic_relevance (Stage 2 cosine).
# CrossEncoder sigmoid score ∈ [0, 1]:
#   >= PRIMARY_THRESHOLD   → focus_label = "primary"
#   >= SECONDARY_THRESHOLD → focus_label = "secondary"
#   <  SECONDARY_THRESHOLD → focus_label = "off-topic"
# =========================
PRIMARY_THRESHOLD: float = 0.55    # CE sigmoid ≥ this → primary  (P-08: recalibrated for BAAI/bge-reranker-base)
SECONDARY_THRESHOLD: float = 0.25  # CE sigmoid ≥ this → secondary (P-08: recalibrated for BAAI/bge-reranker-base)

# CrossEncoder document token budget.
# BAAI/bge-reranker-base: 512 tokens total (query + doc + 3 special tokens).
# With semantic_query ≈ 50–100 tokens + title ≈ 15–25 tokens + 3 special = ~128 overhead,
# abstract budget ≈ 384 tokens minimum ≈ 1536 chars at 4 chars/token.
# Cap at 1500 chars: covers ≈ 90% of arXiv abstracts fully without truncating the query.
_CE_MAX_ABSTRACT_CHARS: int = 1500



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
    # ── Task 37: Retrieval provenance (populated by Task 33 RRF hybrid Stage 1) ──
    retrieval_source: Optional[str] = None   # "bm25_only" | "faiss_only" | "both" (LanceDB full-text | vector)
    bm25_rank: Optional[int] = None          # Full-text search rank within top-K pool (1-indexed)
    faiss_rank: Optional[int] = None         # Vector search rank within top-K pool (1-indexed)
    rrf_score: Optional[float] = None        # Reciprocal Rank Fusion merged score
    # ── Task 38: Artifact signal detection and paper type tagging ──
    has_code: bool = False                   # True if GitHub URL detected in abstract/comments/pdf_url
    has_dataset: bool = False                # True if dataset/benchmark release mentioned
    reproducibility_score: int = 0           # 0-3 count of reproducibility signals
    paper_type_tag: Optional[str] = None     # "New Technique" | "Scale Study" | "New Dataset" | "Survey" | "Evaluation" | "Other"
    # ── Option A: dedicated Stage 3 score field (P-10) ──
    # cross_encoder_score is ONLY written by cross_encoder_rerank (Stage 3 sigmoid of BGE logit).
    # semantic_relevance is ONLY written by Stage 2 (SPECTER2 / MiniLM cosine similarity).
    # Keeping them separate prevents the dual-scale overwrite that caused Phase 4 "no results".
    cross_encoder_score: Optional[float] = None


# =========================
# Utility functions
# =========================

def _get_secret(key: str, default: str = "") -> str:
    """Safely read from st.secrets with os.getenv fallback.
    Handles Cloud Run (env vars only) and Streamlit Cloud (secrets.toml).
    """
    try:
        val = st.secrets.get(key)
        return (val or os.getenv(key, default)).strip().strip('\'"')
    except Exception:
        return os.getenv(key, default).strip().strip('\'"')

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


def ensure_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def get_corpus_dir() -> pathlib.Path:
    """
    Returns the writable directory for data pipeline artifacts.
    Always uses a platform-agnostic temp directory to ensure the app 
    relies exclusively on R2 as the single source of truth.
    """
    # 1. Manual override (mainly for CI/CD or explicit local overrides)
    env_dir = os.environ.get("CORPUS_DATA_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 2. Always fallback to system temp directory (Streamlit Cloud & local usage matches)
    temp_path = pathlib.Path(tempfile.gettempdir()) / "researchagent_corpus"
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


def _check_corpus_freshness():
    """Check R2 for corpus updates at most every 30 min per session; reloads in-place."""
    import boto3
    from botocore.exceptions import ClientError

    now = time.time()
    last_check = st.session_state.get("_freshness_checked_at", 0)
    if now - last_check < 1800: # 1800s = 30 min
        return

    # 2. Get credentials
    access_key  = _get_secret("R2_ACCESS_KEY_ID")
    secret_key  = _get_secret("R2_SECRET_ACCESS_KEY")
    endpoint    = _get_secret("R2_ENDPOINT")
    bucket_name = _get_secret("R2_BUCKET")

    if not all([access_key, secret_key, endpoint, bucket_name]):
        return

    try:
        from botocore.config import Config
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
        
        # 3. HEAD request to check ETag of the metadata file
        response = s3.head_object(Bucket=bucket_name, Key="corpus/build_meta.json")
        new_etag = response.get('ETag', '').strip('"')
        old_etag = st.session_state.get("_corpus_etag")

        if old_etag and new_etag == old_etag:
            st.session_state["_freshness_checked_at"] = now
            return

        # 4. If ETag differs, refresh cached resources
        if old_etag:
            msg = st.info("New corpus data available — refreshing (LanceDB reconnects lazily)...")
            st.cache_resource.clear()  # clears get_lancedb_table so it reconnects on next query
            msg.empty()

        st.session_state["_corpus_etag"] = new_etag
        st.session_state["_freshness_checked_at"] = now

    except Exception:
        # Fail silently: data refresh is non-critical, we don't want to block the search
        pass


def download_corpus_artifacts():
    """
    Startup check: download build_meta.json from R2 for freshness tracking.
    LanceDB reads corpus data lazily directly from R2 — no file downloads needed.
    Runs once per Streamlit session using session_state as a guard.
    """
    if st.session_state.get("_corpus_synced"):
        return

    # Local override: CORPUS_DATA_DIR set → assume local LanceDB already configured
    if os.environ.get("CORPUS_DATA_DIR"):
        st.session_state["_corpus_synced"] = True
        return

    key_id     = _get_secret("R2_ACCESS_KEY_ID")
    access_key = _get_secret("R2_SECRET_ACCESS_KEY")
    endpoint   = _get_secret("R2_ENDPOINT")
    bucket     = _get_secret("R2_BUCKET")

    if not all([key_id, access_key, endpoint, bucket]):
        st.session_state["_corpus_synced"] = True
        return

    corpus_dir = get_corpus_dir()
    corpus_dir.mkdir(parents=True, exist_ok=True)
    meta_path  = corpus_dir / "build_meta.json"

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
        st.session_state["_corpus_synced"] = True

        # Warm-up: pre-load models immediately after startup
        if not st.session_state.get("_models_warmed"):
            try:
                with st.spinner("Pre-loading models (one-time)..."):
                    get_specter2_model()
                    get_cross_encoder_model()
                    get_local_embed_model()
                st.session_state["_models_warmed"] = True
                logging.info("[warmup] All models pre-loaded successfully.")
            except Exception as _e:
                logging.warning("[warmup] Model pre-load failed (non-fatal): %s", _e)

    except Exception as e:
        # Do NOT set _corpus_synced — next rerun retries. LanceDB still connects to
        # R2 directly even without build_meta.json, so this only affects freshness
        # tracking + warm-up timing, not availability.
        logging.warning("[download_corpus_artifacts] build_meta.json sync failed: %s", e)
        st.warning(f"⚠️ Corpus freshness check failed ({e}). Retrying next run.")


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
    terms = [p.strip().lower() for p in parts if p.strip()]
    return terms


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
    selected_venues: List[str]
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
            if selected_category == "Conference":
                if selected_venues and any(sel.lower() in venue for sel in selected_venues):
                    filtered.append(p)
            elif selected_category == "Journal":
                if selected_venues and any(sel.lower() in venue for sel in selected_venues):
                    filtered.append(p)

    return filtered


# =========================
# Venue extraction helpers
# =========================

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
    # Keep it practical; you can extend anytime
    "Computer Science": [
        "cs.AI", "cs.LG", "cs.HC", "cs.CL", "cs.CV", "cs.RO", "cs.IR", "cs.NE", "cs.SE",
        "cs.CR", "cs.DS", "cs.DB", "cs.SI", "cs.MM", "cs.IT", "cs.PF", "cs.MA",
    ],
    # "Statistics": ["stat.ML", "stat.AP", "stat.CO", "stat.TH"],
    # "Mathematics": ["math.OC", "math.ST", "math.IT", "math.PR", "math.NA"],
    # "Physics": ["physics.comp-ph", "physics.data-an", "physics.soc-ph", "physics.optics"],
    # "Quantitative Biology": ["q-bio.QM", "q-bio.NC", "q-bio.BM"],
    # "Quantitative Finance": ["q-fin.MF", "q-fin.ST", "q-fin.CP", "q-fin.TR"],
    # "Electrical Engineering and Systems Science": ["eess.IV", "eess.SP", "eess.SY", "eess.AS"],
    # "Economics": ["econ.EM", "econ.TH"],
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

def extract_venue(comment: str) -> Optional[str]:
    if not comment:
        return None
    c = comment.lower()
    if any(sig in c for sig in NEGATIVE_VENUE_SIGNALS):
        return None
    all_venues = sorted(CONFERENCE_KEYWORDS + JOURNAL_KEYWORDS)
    for venue in all_venues:
        if venue.lower() in c:
            return venue
    return None

DB_FETCH_LIMIT = 20_000  # Hard cap: prevents OOM on Streamlit Cloud with large date windows


def fetch_papers_from_lancedb(
    start_date: date,
    end_date: date,
    category_filter: Optional[str] = None,
    subcats: Optional[List[str]] = None,
) -> List[Paper]:
    """
    Fetch papers from LanceDB on R2. Most-recent first, capped at DB_FETCH_LIMIT.
    Replaces the old SQLite fetch_papers_from_db — no local file download needed.
    """
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

    # Subcat keyword match pushed into the filter (not applied post-LIMIT) so the
    # 20k cap below keeps the most-recent *matching* papers, not the most-recent
    # papers overall with a narrow subcat filter thinning them out afterward.
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

    if "last_prompts" not in st.session_state:
        st.session_state["last_prompts"] = {}
    st.session_state["last_prompts"][label or "default"] = prompt

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if llm_config.provider == "openai":
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
                    st.error("Gemini provider selected but google-genai package is not installed.")
                    st.stop()
                client = genai.Client(api_key=llm_config.api_key)
                response = client.models.generate_content(
                    model=llm_config.model,
                    contents=prompt,
                )
                
                # Handle mixed content
                if hasattr(response, 'candidates') and response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                        texts = []
                        for part in cand.content.parts:
                            if hasattr(part, 'text') and part.text:
                                texts.append(part.text)
                        if texts:
                            return "".join(texts)
                return getattr(response, 'text', "")

            elif llm_config.provider == "groq":
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

@st.cache_resource(show_spinner=False)
def get_local_embed_model() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def embed_texts_local(texts: List[str]) -> List[List[float]]:
    if not texts: return []
    try:
        model = get_local_embed_model()
    except Exception as e:
        st.error(f"Local embedding model error: {e}")
        st.stop()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2): return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1)
    norm2 = sum(b * b for b in vec2)
    if norm1 == 0.0 or norm2 == 0.0: return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


@st.cache_resource(show_spinner=False)
def get_lancedb_table():
    """
    Connect to LanceDB (R2 via env vars, or local CORPUS_DATA_DIR) and return the papers table.
    Cached per Streamlit session — cleared by _check_corpus_freshness on corpus update.
    """
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

@st.cache_resource(show_spinner=False)
def get_specter2_model():
    """
    Load SPECTER2 base + adhoc_query adapter for asymmetric scientific retrieval.

    Query side  → adhoc_query adapter (short free-text query)
    Paper side  → encode as: title + [SEP] + abstract  (handled in specter2_vector_rerank)

    Returns (model, tokenizer) on success, (None, None) on any failure so the
    caller can fall back to the MiniLM pre-computed embedding path.
    """
    try:
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        import torch

        # NOTE: timeout is controlled via HF_HUB_DOWNLOAD_TIMEOUT env var (set at top of file).
        # Do NOT pass timeout= to from_pretrained — adapters lib forwards it to
        # BertAdapterModel.__init__() which rejects it with TypeError.
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained(
            "allenai/specter2_base",
            torch_dtype=torch.float16,
        )

        # Load and activate the adhoc_query adapter for query-to-paper retrieval
        model.load_adapter(
            "allenai/specter2_adhoc_query",
            source="hf",
            load_as="specter2_adhoc_query",
            set_active=True,
        )
        model.set_active_adapters("specter2_adhoc_query")

        # Verify adapter is active
        try:
            active_adapters = model.active_adapters
            if "specter2_adhoc_query" not in active_adapters:
                print(f"[SPECTER2] Warning: Adapter not active. Active adapters: {active_adapters}")
                # Try to set it active again
                model.set_active_adapters("specter2_adhoc_query")
        except Exception as e:
            print(f"[SPECTER2] Warning: Could not verify adapter activation: {e}")

        # load_adapter() adds adapter weights in float32 regardless of the base
        # model's dtype — recast the whole model (base + adapter) so forward
        # passes don't hit a Half/Float matmul mismatch.
        model = model.to(torch.float16)
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

    Encoding protocol (correct SPECTER2 asymmetric format):
      - Query : encode live with adhoc_query adapter (~0.3s, one forward pass)
      - Paper : look up SPECTER2 proximity-adapter vectors already fetched from LanceDB
                by _lancedb_hybrid_stage1 (paper_vectors dict).
                Only papers absent from that dict fall back to a live adhoc_query pass,
                which is a small minority in practice.

    Fast path:   paper_vectors provided → ~0.05s for 1200 papers
    Slow path:   paper_vectors empty → live SPECTER2 encode for all papers (~60s/1200)
    Fallback:    SPECTER2 model unavailable → return [] so caller uses MiniLM path

    Sets p.semantic_relevance on each paper.
    """
    if not papers:
        return []

    model, tokenizer = get_specter2_model()
    if model is None or tokenizer is None:
        return []   # Caller falls back to minilm_vector_rerank

    import torch

    # Ensure adhoc_query adapter active for query encoding
    try:
        model.set_active_adapters("specter2_adhoc_query")
    except Exception as e:
        print(f"[SPECTER2] Warning: Could not set active adapter: {e}")

    def _encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
                return_token_type_ids=False,
            )
            with torch.no_grad():
                output = model(**inputs)
            vecs = output.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    try:
        # Always encode query live (not precomputed)
        q_vecs = _encode_batch([query_brief])
        q_vec = q_vecs[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # ── Fast path: use vectors fetched from LanceDB in Stage 1 ──
        have_lancedb_vecs = bool(paper_vectors)

        if have_lancedb_vecs:
            # Split papers into indexed (fast lookup) and unindexed (live encode)
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

            # Vectorized cosine for indexed papers — single numpy op
            if indexed_papers:
                indexed_vecs = np.vstack(indexed_vecs_list).astype("float32")
                norms = np.linalg.norm(indexed_vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                indexed_vecs = indexed_vecs / norms
                indexed_sims = indexed_vecs @ q_vec  # (N_indexed,)
            else:
                indexed_sims = np.array([], dtype="float32")

            # Live encode only the minority not in the precomputed index
            if unindexed_papers:
                sep = tokenizer.sep_token or " "
                unindexed_texts = [
                    p.title + sep + (p.abstract if p.abstract else "")
                    for p in unindexed_papers
                ]
                unindexed_vecs = _encode_batch(unindexed_texts)
                norms = np.linalg.norm(unindexed_vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                unindexed_vecs = unindexed_vecs / norms
                unindexed_sims = unindexed_vecs @ q_vec
            else:
                unindexed_sims = np.array([], dtype="float32")

            # Merge, score, sort
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
            # ── Slow path: no precomputed embeddings — encode all live ──────
            sep = tokenizer.sep_token or " "
            paper_texts = [
                p.title + sep + (p.abstract if p.abstract else "")
                for p in papers
            ]
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
    if not papers: return []
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

    Returns (stage1_papers, paper_vectors).
    paper_vectors: {arxiv_id: list[float]} — SPECTER2 proximity vectors fetched from LanceDB.
    These are passed to specter2_vector_rerank as the Stage 2 fast path, replacing the old
    embeddings.npy array. Only vectors for papers with has_embedding=True are included.

    Search scoped globally then intersected with the `papers` pool (already date-filtered).
    search_k over-fetches to compensate for pool-intersection loss.
    """
    if not papers:
        return [], {}

    paper_ids  = {p.arxiv_id for p in papers}
    paper_dict = {p.arxiv_id: p for p in papers}
    fts_ranks:    Dict[str, int]  = {}
    vec_ranks:    Dict[str, int]  = {}
    paper_vectors: Dict[str, list] = {}

    search_k = min(top_k * 5, 6000)

    # Derive pool date range to scope global searches — critical for "All Time" mode
    # where DB_FETCH_LIMIT caps the pool at 20k out of 385k total papers.
    # Without scoping, only ~5% of global search results land in the pool.
    pool_date_where: Optional[str] = None
    if papers:
        dates = [p.submitted_date.strftime("%Y-%m-%d") for p in papers if p.submitted_date]
        if dates:
            min_date, max_date = min(dates), max(dates)
            pool_date_where = (
                f"submitted_date >= '{min_date}' AND submitted_date <= '{max_date}T23:59:59'"
            )

    # QIL quality_modifier pre-filter (year/citation_count), ANDed alongside the date scope.
    scope_where_parts = [p for p in (pool_date_where, quality_where) if p]
    scope_where = " AND ".join(f"({p})" for p in scope_where_parts) if scope_where_parts else None

    # ── FTS search (Tantivy, searches all FTS-indexed fields: title + abstract) ──
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

    # ── Vector search (IvfHnswSq cosine, SPECTER2 proximity vectors) ─────────
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

    # ── RRF merge (k=60, equal weights for FTS and vector) ────────────────────
    _k = 60
    union_ids = set(fts_ranks.keys()) | set(vec_ranks.keys())
    n_fts, n_vec = len(fts_ranks), len(vec_ranks)

    scored: List[tuple] = []
    for aid in union_ids:
        rrf = (
            1.0 / (_k + fts_ranks.get(aid, n_fts + 1))
            + 1.0 / (_k + vec_ranks.get(aid, n_vec + 1))
        )
        scored.append((rrf, aid))
    scored.sort(reverse=True)

    result_papers: List[Paper] = []
    for rrf_score, aid in scored[:top_k]:
        p = paper_dict.get(aid)
        if p:
            p.bm25_rank  = fts_ranks.get(aid)
            p.faiss_rank = vec_ranks.get(aid)
            p.rrf_score  = rrf_score
            p.retrieval_source = (
                "both"       if aid in fts_ranks and aid in vec_ranks
                else "bm25_only"  if aid in fts_ranks
                else "faiss_only"
            )
            result_papers.append(p)

    return result_papers, paper_vectors


@st.cache_resource(show_spinner=False)
def get_cross_encoder_model():
    try:
        from sentence_transformers import CrossEncoder
        import torch
        # P-07: Swapped from ms-marco-MiniLM-L-6-v2 to BAAI/bge-reranker-base
        # bge-reranker-base is trained on broader multilingual+scientific corpus;
        # consistently outperforms ms-marco on BEIR scientific subsets (SciFact, TREC-COVID).
        st.info("Loading CrossEncoder model (BAAI/bge-reranker-base) for precision re-ranking (first run only)…")
        return CrossEncoder("BAAI/bge-reranker-base", automodel_args={"torch_dtype": torch.float16})

    except Exception as e:
        print(f"CrossEncoder load error: {e}")
        return None

def cross_encoder_rerank(papers: List[Paper], query_brief: str, n3: int = 150) -> List[Paper]:
    """
    P-10: Writes sigmoid(logit) exclusively to p.cross_encoder_score.
    Does NOT overwrite p.semantic_relevance (Stage 2 cosine — preserved for display + fallback).
    When model is unavailable, returns papers[:n3] with cross_encoder_score=None so
    scibert_classify_papers can detect the failure and use cosine-based heuristic fallback.

    Token budget: BAAI/bge-reranker-base has a 512-token window shared by query and document.
    Abstract is capped at _CE_MAX_ABSTRACT_CHARS (1500 chars) so the query is never truncated
    and the maximum abstract content is preserved within the remaining token budget.
    Caller passes semantic_query (~50 tokens) not raw query_brief to maximise abstract coverage.
    """
    if not papers: return []
    model = get_cross_encoder_model()
    if not model:
        # Model unavailable — cross_encoder_score stays None; classifier uses heuristic fallback
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
            # P-10: dedicated field only — semantic_relevance (Stage 2 cosine) is NOT touched
            p.cross_encoder_score = 1 / (1 + math.exp(-score_float))
            scored.append((score_float, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(n3, len(scored))
        return [p for _, p in scored[:k]]
    except Exception as e:
        print(f"CrossEncoder predict error: {e}")
        # cross_encoder_score stays None on failure — heuristic fallback kicks in at Phase 4
        return papers[:n3]


def _st_emit(msg: str, level: str = "write") -> None:
    getattr(st, level)(msg)


def select_embedding_candidates(
    papers: List[Paper],
    query_brief: str,
    llm_config: Optional[LLMConfig] = None,
    max_candidates: int = 150,
    emit: Callable[[str, str], None] = _st_emit,
    qil_cache: Optional[dict] = None,
) -> List[Paper]:
    """
    4-stage hybrid search pipeline:
      Stage 0 — QIL decomposes brief → semantic_query, bm25_keywords, intent
      Stage 1 — LanceDB full-text + vector search → RRF → adaptive top-K
      Stage 2 — SPECTER2 vector rerank → top-400 (MiniLM fallback)
      Stage 3 — CrossEncoder precision rerank → top-150
    Graceful degradation: full-text-only, vector-only, or full-pool fallback.

    No direct Streamlit calls — `emit(msg, level)` defaults to st.write/info/warning
    but can be swapped for a plain collector in tests or non-Streamlit callers.
    `qil_cache` defaults to a fresh per-call dict; pass a persistent dict (e.g.
    st.session_state) to keep QIL cross-call caching.
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
            # Cache: avoid redundant LLM calls for identical queries within a session
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
                "llm_groq":       "LLM/Groq",
                "llm_openrouter": "LLM/OpenRouter",
                "rules":          "Rules",
            }.get(sq.source, sq.source.upper())
            emit(
                f"⏱ Stage 0 (QIL/{_source_label}): intent=`{sq.intent}` · "
                f"quality=`{sq.quality_modifier}` · "
                f"keywords=`{', '.join(sq.bm25_keywords[:4])}{'...' if len(sq.bm25_keywords) > 4 else ''}`"
            )
        except Exception as _qil_err:
            print(f"[QIL] query analysis failed: {_qil_err} — using raw brief")
            sq = None

    bm25_query     = sq.bm25_query_string if sq and sq.bm25_keywords else None
    semantic_query = sq.semantic_query    if sq and sq.semantic_query  else None

    # ─── Stage 1: LanceDB FTS + vector search → manual RRF ───────────────
    # P-04: Adaptive top-K — 7% of pool, capped at 1200
    adaptive_k = min(int(len(papers) * 0.07), 1200)
    adaptive_k = max(adaptive_k, 50)
    emit(f"⏱ Stage 1: LanceDB FTS + vector → RRF merge (adaptive_k={adaptive_k})...")

    # Generate query vector once for both Stage 1 vector search and Stage 2 fast path
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
                "— using full pool as fallback.",
                "info",
            )
            stage1_papers = papers
        else:
            n_both     = sum(1 for p in stage1_papers if p.retrieval_source == "both")
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
    # hard_filters.not_terms extracted by QIL (LLM or rules) from the research brief.
    # Applied here (post-Stage 1, pre-Stage 2) so explicit exclusions don't pollute
    # SPECTER2/CrossEncoder ranking and don't need a second pass later.
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
    # paper_vectors from Stage 1 replaces the old embeddings.npy array.
    # Only the ~top_k vectors are in RAM (per-query) instead of the full 1.2 GB matrix.
    stage2_query = semantic_query if semantic_query else query_brief
    have_stage1_vecs = bool(paper_vectors)
    emit(
        f"⏱ Stage 2: SPECTER2 semantic reranking "
        f"({'LanceDB vector fast-path' if have_stage1_vecs else 'live-encode'})..."
    )
    stage2_papers = specter2_vector_rerank(
        stage1_papers,
        stage2_query,
        n2=200,
        paper_vectors=paper_vectors,
    )
    if stage2_papers:
        emit(
            f"✅ SPECTER2 Stage 2: {len(stage2_papers)} candidates selected "
            f"(scientific asymmetric retrieval)."
        )
    else:
        emit(
            "⚠️ SPECTER2 unavailable — falling back to MiniLM for Stage 2. "
            "Retrieval quality may be reduced. Check that `allenai/specter2` is installed.",
            "warning",
        )
        stage2_papers = minilm_vector_rerank(
            stage1_papers,
            stage2_query,
            n2=200,
        )
        emit(f"✅ MiniLM Stage 2 fallback: {len(stage2_papers)} candidates selected.")

    # ─── Stage 3: CrossEncoder Precision Rerank ───────────────────────
    # P-00: Cross-encoder pairs against semantic_query for a cleaner signal.
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
    to the top-2 most relevant sentences. This provides auditability and explains
    why the paper matched the semantic intent.
    """
    if not papers: 
        return papers

    try:
        model = get_local_embed_model()
        q_vec = model.encode([query_brief], normalize_embeddings=True)[0]
    except Exception as e:
        print(f"Failed to load embed model for abstract highlights: {e}")
        return papers

    import re
    # re.split to split on sentence boundaries
    sent_regex = re.compile(r'(?<=[.!?])\s+')

    for p in papers:
        if not p.abstract:
            continue
        
        sentences = [s.strip() for s in sent_regex.split(p.abstract) if len(s.strip()) >= 15]
        if not sentences:
            p.semantic_reason = "No extractable sentences found in abstract."
            continue
        
        try:
            s_vecs = model.encode(sentences, normalize_embeddings=True)
            # Dot product works because vectors are normalized
            scores = [float(np.dot(vec, q_vec)) for vec in s_vecs]
            
            # Sort sentences by score descending
            scored_sentences = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
            
            # Take top 2
            top_sents = [s for _, s in scored_sentences[:2]]
            
            if len(top_sents) == 1:
                p.semantic_reason = f"Matched: '{top_sents[0][:120]}...'"
            else:
                p.semantic_reason = f"Matched: '{top_sents[0][:120]}...' | '{top_sents[1][:120]}...'"
                
        except Exception as e:
            print(f"Highlight extraction error for {p.arxiv_id}: {e}")
            pass

    return papers


# =========================
# Task 38 — Artifact signal detection and paper type tagging
# =========================

def enrich_paper_signals(papers: List[Paper]) -> List[Paper]:
    """
    Called after Stage 4 to set artifact signals (has_code, has_dataset,
    reproducibility_score, paper_type_tag) based on fast regex/string regex.
    """
    for p in papers:
        abstract_lower = p.abstract.lower() if p.abstract else ""
        pdf_lower = p.pdf_url.lower() if p.pdf_url else ""
        
        # has_code
        if "github.com/" in abstract_lower or "github.com/" in pdf_lower:
            p.has_code = True
            
        # has_dataset
        dataset_keywords = ["new dataset", "we release", "we introduce a benchmark", "data collection", "we collected"]
        if any(kw in abstract_lower for kw in dataset_keywords):
            p.has_dataset = True
            
        # reproducibility_score
        score = 0
        if "ablation" in abstract_lower:
            score += 1
        if "code available" in abstract_lower or "github.com/" in abstract_lower:
            score += 1
        if "reproducib" in abstract_lower:
            score += 1
        p.reproducibility_score = score
        
        # paper_type_tag
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
    P-10: CrossEncoder threshold classification with explicit heuristic fallback.

    Primary path (CE available):
      Reads p.cross_encoder_score (Stage 3 sigmoid of BGE-reranker-base logit).
      Thresholds: PRIMARY_THRESHOLD=0.55, SECONDARY_THRESHOLD=0.25.

    Fallback path (CE unavailable — model failed to load or predict raised):
      p.cross_encoder_score is None for all papers.
      Delegates to heuristic_classify_papers_free(), which uses rank-based
      top-30% = primary over p.semantic_relevance (Stage 2 cosine).
      This guarantees at least min(n, max(10, 30% of n)) primary papers,
      preventing the "no results" failure when CrossEncoder is unavailable.
    """
    ce_available = any(p.cross_encoder_score is not None for p in papers)

    if not ce_available:
        # No CE scores — cosine similarities in semantic_relevance; use rank-based heuristic
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
    if not candidates: return candidates
    ranked = sorted(candidates, key=lambda p: p.semantic_relevance or 0.0, reverse=True)
    n = len(ranked)
    if n == 0: return ranked
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

RECENCY_BONUS_MAX = 15.0  # ponytail: additive cap for "recent" quality_modifier, tune if it over/under-shoots

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
    target_papers: List[Paper], llm_config: LLMConfig, batch_size: int = 8,
    quality_modifier: str = "any",
) -> List[Paper]:
    """MONEYBALL PREDICTOR: Hybrid Author Data + Custom LLM Narrative."""
    if not target_papers: return target_papers

    weights = resolve_moneyball_weights(quality_modifier)
    
    # Read S2 key from st.secrets (Streamlit Cloud) with os.getenv fallback (local/.env)
    s2_key = _get_secret("S2_API_KEY")
    progress_bar = st.progress(0)
    
    for i, p in enumerate(target_papers):
        # 1. Get Hard Data (Fame Signal)
        max_auth_cites = get_s2_citation_stats(p, s2_key)
        
        # LOGIC: Check if we have data or if the paper is too new
        is_fresh = False
        try:
            # DETERMINISTIC CHECK: Purely based on date, ignoring API flakiness result
            days_old = (datetime.now().date() - p.submitted_date.date()).days
            if days_old <= 5: is_fresh = True
        except: pass

        if max_auth_cites > 0:
            # We have real data -> Standard Model
            h1_fame = min(math.log(max_auth_cites + 1) * 8, 95)
            fame_label = "real"
        elif is_fresh:
            # Too New + No Data -> DO NOT SCORE
            fame_label = "too_new"
            h1_fame = 0.0 # Sentinel
        else:
            # Old + No Data -> Likely obscure
            h1_fame = 0.0
            fame_label = "none"

        if not s2_key: time.sleep(0.3) 
        
        # 2. Calculate Hype/Sniper (Python Heuristics)
        t_lower = p.title.lower()
        h2_hype = 0
        if "benchmark" in t_lower or "dataset" in t_lower: h2_hype += 50
        if "survey" in t_lower: h2_hype += 40
        if "llm" in t_lower: h2_hype += 10
        
        h3_sniper = 0
        if "benchmark" in t_lower: h3_sniper += 50
        niche = ["lidar", "3d", "audio", "wireless", "agriculture", "traffic", "physics"]
        if any(n in t_lower for n in niche): h3_sniper -= 20
        
        # 3. LLM Call: Get Score AND Custom Narrative
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
            "The paper proposes a specific contribution to the field."
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

        # 4. Calculate Final Score
        if fame_label == "too_new":
            # SENTINEL VALUE: -1.0 means "Unrated"
            p.predicted_citations = -1.0
        else:
            score = (h1_fame * weights['weight_fame'] +
                     h2_hype * weights['weight_hype'] +
                     h3_sniper * weights['weight_sniper'] +
                     h4_utility * weights['weight_utility'])
            if quality_modifier == "recent":
                score += compute_recency_score(p.submitted_date) * RECENCY_BONUS_MAX
            p.predicted_citations = score

        # 5. Construct Final Narrative (3 Bullets)
        final_bullets = []
        
        # Bullet 1: Author Context
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
        
        progress_bar.progress((i + 1) / len(target_papers))
        
    progress_bar.empty()
    return target_papers


def compute_recency_score(submitted_date: datetime, max_days: int = 30) -> float:
    """
    Returns a recency score in [0, 1], where newer papers score higher.
    """
    try:
        days_old = max((datetime.now().date() - submitted_date.date()).days, 0)
    except Exception:
        return 0.0

    if days_old >= max_days:
        return 0.0

    return 1.0 - (days_old / max_days)

def assign_heuristic_citations_free(papers: List[Paper]) -> List[Paper]:
    if not papers: return papers
    scores = [(p.llm_relevance_score or 0.0) * 0.7 + (p.semantic_relevance or 0.0) * 0.3 for p in papers]
    if not scores: return papers
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


# =========================
# Streamlit UI
# =========================

PIPELINE_DESCRIPTION_MD = """
#### 1. Describe what you want

You write a short research brief in natural language about the kind of work you care about, and optionally what you are not interested in. If you leave both fields empty, the agent switches to a global mode and just looks for the most impactful recent Computer Science papers overall.

#### 2. The agent searches a curated local corpus

Instead of fetching papers live, the agent queries a pre-built local library of 40,000+ papers harvested from arXiv and Semantic Scholar. This library is refreshed on a schedule via a pipeline and, when configured, synced from cloud storage (Cloudflare R2) on startup — so you always get fast, up-to-date results without depending on external API availability at search time.

#### 3. The agent picks candidate papers using 3-stage hybrid search

Retrieval is a three-step funnel designed to be both fast and accurate:

- **Stage 1 — Hybrid Recall (LanceDB FTS + Vector RRF):** Quickly narrows the corpus using reciprocal rank fusion of full-text search and SPECTER2 vector search from LanceDB.
- **Stage 2 — Semantic Rethink (SPECTER2):** A powerful document-level embedding model (SPECTER2) re-evaluates the candidate pairs to refine the rankings.
- **Stage 3 — Precision Reranking (CrossEncoder):** A deep cross-attention model does a final comparison between your brief and each candidate, selecting the best papers to pass to the next stage.

#### The agent filters by venue (Optional)

If you selected a venue filter (e.g. "NeurIPS only" or "All Journals"), the agent applies it **after** the hybrid search. This ensures the agent first identifies the most semantically relevant papers from the entire corpus, and then narrows them down to your preferred venues.

#### 4. The agent judges how relevant each paper is

A CrossEncoder model (`BAAI/bge-reranker-base`) scores each candidate against the research brief. Papers scoring ≥ 0.55 are labelled **primary**; ≥ 0.25 are **secondary**; below 0.25 are filtered out. No LLM tokens are used at this step.

#### 5. The agent builds a citation impact set

The agent builds a set of papers to send to the citation impact step:

- It keeps all **primary** papers.
- If there are fewer than about 20, it tops up with the strongest **secondary** papers until it reaches roughly 20, when possible.
- In global mode, all candidates are used.

#### 6. The agent computes 1-year citation impact scores

- In **LLM API mode** (OpenAI, Gemini, or Groq), a model estimates a 1-year citation impact score for each paper using author citation data from Semantic Scholar and an LLM-generated narrative, and provides short explanations.
- In **free local mode**, the agent derives a citation impact score from the relevance signals and uses that to rank papers.

These scores are heuristic impact signals and are best used for ranking within this batch, not as ground truth.

**Note on New Papers:** Papers less than 5 days old often lack citation data in Semantic Scholar. These are marked as **"Too new for impact score"** and ranked purely by their relevance to your query.

#### 7. The agent ranks, summarizes, and saves results

The agent ranks papers, always showing **primary** papers first, then secondary ones. For the top N that you choose, it shows metadata, relevance signals, and links to arXiv and the PDF. In LLM API mode it also adds plain English summaries. All artifacts and a markdown report are saved in a project folder under `~/arxiv_ai_digest_projects/project_<timestamp>`, and you can download everything as a ZIP.
"""

def main():
    st.set_page_config(
        page_title="Research Agent",
        layout="wide",
    )

    # Startup sync from Cloudflare R2
    download_corpus_artifacts()

    # ===== FOOTER (injected early so no early return can skip it) =====
    # Uses CSS to push itself to the bottom of the viewport when content is short,
    # and sits naturally at the end of content when the page is long.
    st.markdown(
        """
        <style>
        /* Make the main Streamlit container fill at least the full viewport height */
        .stMainBlockContainer {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        /* Push footer to the bottom by giving the content area above it flex-grow */
        .block-container {
            flex-grow: 1;
        }
        .page-footer {
            text-align: center;
            color: gray;
            font-size: 0.85rem;
            padding: 2rem 0 1rem 0;
            border-top: none;
            margin-top: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # We define a function to call at every exit point
    def render_footer():
        st.markdown(
            """
            <div class="page-footer">
                &copy; The Benevolent Bandwidth Foundation, Inc. &middot; Massachusetts Nonprofit Corporation. All rights reserved.<br>
                Built with ❤️ for humanity
            </div>
            """,
            unsafe_allow_html=True,
        )
    # ===== END FOOTER SETUP =====

    _main_body()
    render_footer()


def _main_body():
    st.title("🔎 Research Agent")

    
    st.write(
        """Too many important papers get lost in the noise. Most researchers and practitioners cannot reliably scan what is new recently in their area, find truly promising work, and trust that they did not miss something big."""
        " This agent helps with this problem by finding, ranking, and explaining recent AI papers on arxiv.org."
        " Run time can be lengthy if you select a large time window or a large backend LLM. Patience is a virtue for good things to come!"
    )

    # Sidebar
    with st.sidebar:
        st.header("🧠 Research Brief")

        research_brief_default = (
            "I am interested in papers whose MAIN contribution is about recommendation systems: "
            "for example, new model architectures, training strategies, evaluation methods, user or item modeling, "
            "or personalization techniques for recommenders.\n\n"
            "I especially care about work where recommendation is the central focus, not just a side example."
        )

        research_brief = st.text_area(
            "What kinds of topics are you looking for?",
            value=research_brief_default,
            height=200,
            help="Describe your research interest in natural language. Focus on what the main contribution "
                 "of the papers should be. If you leave this and the next box empty, the agent will perform "
                 "a global digest of recent Computer Science papers."
        )

        not_looking_for = st.text_area(
            "What are you NOT looking for? (optional)",
            value="Generic LLM papers that only list recommendation as one of many downstream tasks, "
                  "or papers that focus purely on language modeling, math reasoning, or scaling without "
                  "a clear recommendation specific contribution.",
            height=120,
        )

        #----Category selection UI-----
        st.markdown("### 🧩 arXiv Category")

        main_cat = st.selectbox(
            "Main category",
            list(ARXIV_CATEGORIES.keys()),
            index=0  # default to "Computer Science"
        )

        if main_cat == "All":
            # optional: allow multiselect of mains, but simplest is "All categories"
            subcats = []
            st.caption("Using all available categories.")
        else:
            subcats = st.multiselect(
                "Subcategory (choose one or more)",
                options=ARXIV_CATEGORIES[main_cat],
                default=["cs.AI", "cs.LG", "cs.HC"] if main_cat == "Computer Science" else [],
                format_func=lambda x: f"{ARXIV_CODE_TO_NAME.get(x, x)} ({x})",
                help="If you choose none, we'll use ALL subcategories from the selected main category."
            )

        # --- Venue Filtering UI ---

        st.markdown("### 🏷 Venue Filter")

        venue_filter_type = st.selectbox(
            "Filter by venue",
            ["None", "All Conferences", "All Journals", "Specific Venue"],
            index=0
        )

        selected_category = None
        selected_venues = []

        if venue_filter_type == "Specific Venue":

            # First dropdown: choose category
            selected_category = st.selectbox(
                "Select type:",
                ["Conference", "Journal"]
            )

            # Depending on category show MULTISELECT
            if selected_category == "Conference":
                options = sorted(CONFERENCE_KEYWORDS)
            else:
                options = sorted(JOURNAL_KEYWORDS)

            selected_venues = st.multiselect(
                f"Select {selected_category.lower()}(s):",
                options=options
            )

        

        date_option = st.selectbox("Date Range", ["Last 3 Days", "Last Week", "Last Month", "All Time"], index=2)

        st.markdown("### ⭐ Top N Highlight")
        top_n = st.slider(
            "How many top papers to highlight?",
            1, 10, 3
        )

        st.markdown("### 🔌 Provider")

        provider_label_groq = "Free (Groq — no CC required)"
        provider_label_openai = "OpenAI (API key required)"
        provider_label_gemini = "Gemini (API key required)"
        provider_label_local = "Local Dev (Heuristics only)"

        provider_choice = st.radio(
            "Choose provider",
            [provider_label_groq, provider_label_openai, provider_label_gemini, provider_label_local],
            index=0,
        )

        if provider_choice == provider_label_openai:
            provider = "openai"
        elif provider_choice == provider_label_gemini:
            provider = "gemini"
        elif provider_choice == provider_label_local:
            provider = "free_local"
        else:
            provider = "groq"

        if provider == "openai":
            api_base = "https://api.openai.com/v1"
        elif provider == "gemini":
            api_base = "https://generativelanguage.googleapis.com"
        else:
            api_base = ""

        if provider == "openai":
            st.markdown("### 🤖 OpenAI Settings")
            api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            st.caption(
                "Your API key is used only in memory for this session, is never written to disk, "
                "and is never shared with anyone or any service other than OpenAI's API. "
                "When your session ends, the key is cleared from the app's state."
            )

            openai_models = [
                "gpt-5.2",
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o-mini",
                "gpt-4o",
                "o1",
            ]
            model_choice = st.selectbox(
                "OpenAI Chat model (for classification & citation impact scoring)",
                openai_models,
                index=0,
            )
            if model_choice == "Custom":
                model_name = st.text_input(
                    "Custom OpenAI Chat model name",
                    value="gpt-4.1-mini",
                    help="Example: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini, o1, etc."
                )
            else:
                model_name = model_choice

            embedding_model_name = "allenai/specter2 (local, MiniLM fallback)"
            st.caption(
                f"Embeddings: `{embedding_model_name}`. Retrieval always runs on local "
                "SPECTER2/MiniLM embeddings — OpenAI is used for classification & citation scoring only."
            )

        elif provider == "gemini":
            st.markdown("### 🌌 Gemini Settings")
            api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
            st.caption(
                "Use an API key from Google AI Studio or Vertex AI for the Gemini API. "
                "The key is kept only in memory for this session and is never written to disk."
            )

            # Updated Gemini models list including Gemini 3 Preview
            gemini_models = [
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash-exp",
            ]
            gemini_choice = st.selectbox(
                "Gemini Chat model (for classification & citation impact scoring)",
                gemini_models,
                index=0,
            )
            if gemini_choice == "Custom":
                model_name = st.text_input(
                    "Custom Gemini model name",
                    value="gemini-3-pro-preview",
                    help="Use the model identifier shown in Google AI Studio, for example `gemini-3-pro-preview`."
                )
            else:
                model_name = gemini_choice

            embedding_model_name = "allenai/specter2 (local, MiniLM fallback)"
            st.caption(
                f"Embeddings: `{embedding_model_name}`. Retrieval always runs on local "
                "SPECTER2/MiniLM embeddings — Gemini is used for classification & citation scoring only."
            )

        elif provider == "groq":
            st.markdown("### ⚡ Groq Settings")
            api_key = st.text_input("Groq API Key (Optional)", type="password", value=os.getenv("GROQ_API_KEY", ""))
            st.caption(
                "Groq API keys are free. No credit card needed. Get yours at [https://console.groq.com](https://console.groq.com). "
                "If omitted, the classification will still run using CrossEncoder, but the narrative score will fallback to a default."
            )

            groq_models = [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "qwen-qwq-32b",
                "gemma2-9b-it",
            ]

            model_choice = st.selectbox(
                "Groq Model",
                groq_models,
                index=0,
            )

            if model_choice == "Custom":
                model_name = st.text_input(
                    "Custom Groq Model Name",
                    value="llama-3.3-70b-versatile",
                )
            else:
                model_name = model_choice

            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            st.caption("Using free local MiniLM embeddings (Groq does not provide embeddings).")

        else:
            api_key = ""
            model_name = "heuristic-free-local"
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            st.caption(
                f"Embeddings (local): `{embedding_model_name}`.\n"
                "Classification and citation impact scoring use simple heuristics. No API key or external calls."
            )

        run_clicked = st.button("🚀 Run Pipeline")

    if run_clicked:
        _check_corpus_freshness()
        st.session_state["hide_pipeline_description"] = True

    hide_desc = st.session_state.get("hide_pipeline_description", False)

    if hide_desc:
        with st.expander("Show full pipeline description", expanded=False):
            st.markdown(PIPELINE_DESCRIPTION_MD)
    else:
        st.markdown(PIPELINE_DESCRIPTION_MD)

    # Mode and query brief
    brief_text = research_brief.strip()
    not_text = not_looking_for.strip()

    if not brief_text and not not_text:
        mode = "global"
        query_brief = (
            "User wants to see the most impactful recent AI, ML, and HCI papers in cs.AI, cs.LG, and cs.HC, "
            "without any additional topical filter."
        )
    elif not brief_text and not_text:
        mode = "broad_not_only"
        rb_prompt = (
            "User is broadly interested in recent AI, ML, and HCI work in cs.AI, cs.LG, and cs.HC."
        )
        query_brief = build_query_brief(rb_prompt, not_looking_for)
    else:
        mode = "targeted"
        query_brief = build_query_brief(research_brief, not_looking_for)

    params = {
        "research_brief": research_brief.strip(),
        "not_looking_for": not_looking_for.strip(),
        "date_option": date_option,
        "top_n": top_n,
        "model_name": model_name,
        "provider": provider,
        "venue_filter_type": venue_filter_type,
        "selected_category": selected_category,
        "selected_venues": selected_venues,
        "main_cat": main_cat,
        "subcats": subcats,
    }

    if "last_params" not in st.session_state:
        st.session_state["last_params"] = params.copy()

    if params != st.session_state["last_params"] and not run_clicked:
        for key in [
            "current_papers",
            "candidates",
            "used_papers",
            "used_label",
            "ranked_papers",
            "topN",
            "project_folder",
            "timestamp",
            "zip_bytes",
            "config",
            "mode",
            "current_start",
            "current_end",
            "plain_summaries",
        ]:
            st.session_state.pop(key, None)
        st.session_state["last_params"] = params.copy()
        st.info("Sidebar settings changed. Click **Run Pipeline** to generate new results.")
        return

    if run_clicked:
        st.session_state["last_params"] = params.copy()

    if provider == "openai":
        if not api_key or not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your OpenAI API key and chat model name are required to run in OpenAI mode.")
                return
    elif provider == "gemini":
        if not api_key or not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your Gemini API key and model name are required to run in Gemini mode.")
                return
    elif provider == "groq":
        if not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your Groq model name is required to run in Groq mode.")
                return
    else:
        api_key = api_key or ""
        model_name = model_name or "heuristic-free-local"

    llm_config = LLMConfig(
        api_key=api_key or "",
        model=model_name,
        api_base=api_base,
        provider=provider,
    )
    
    active_llm_config = (llm_config if api_key.strip() else None) if provider in ("openai", "gemini", "groq") else None

    try:
        current_start, current_end = get_date_range(date_option)
    except ValueError as e:
        st.error(str(e))
        return

    st.session_state["mode"] = mode
    st.session_state["current_start"] = current_start
    st.session_state["current_end"] = current_end

    if not run_clicked and "ranked_papers" not in st.session_state:
        st.info("Fill in your research brief and settings in the sidebar, then click **Run Pipeline**.")
        return

    # 1. Project setup
    st.subheader("1. Project Setup")

    root_base_default = os.path.expanduser("~/arxiv_ai_digest_projects")
    base_folder = ensure_folder(root_base_default)

    if run_clicked or "project_folder" not in st.session_state or "timestamp" not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = os.path.join(base_folder, f"project_{timestamp}")
        project_folder = ensure_folder(project_folder)
        st.session_state["project_folder"] = project_folder
        st.session_state["timestamp"] = timestamp
    else:
        project_folder = st.session_state["project_folder"]
        timestamp = st.session_state["timestamp"]

    st.write(f"Project folder: `{project_folder}`")

    config = {
        "mode": mode,
        "query_brief": query_brief,
        "research_brief": research_brief,
        "not_looking_for": not_looking_for,
        "date_option": date_option,
        "current_start": str(current_start),
        "current_end": str(current_end),
        "project_folder": project_folder,
        "created_at": datetime.now().isoformat(),
        "llm_model": model_name,
        "llm_api_base": api_base,
        "embedding_model": embedding_model_name,
        "llm_provider": (
            "OpenAI" if provider == "openai"
            else "Gemini" if provider == "gemini"
            else "Groq" if provider == "groq"
            else "FreeLocalHeuristic"
        ),
        "top_n": top_n,
        "min_for_prediction": MIN_FOR_PREDICTION,
    }
    st.session_state["config"] = config
    save_json(os.path.join(project_folder, "config.json"), config)

    # 2. Fetch current papers from local corpus
    st.subheader("2. Fetch Current Papers from Corpus")

    if run_clicked or "current_papers" not in st.session_state:
        with st.spinner("Loading papers from LanceDB by date window..."):
            current_papers = fetch_papers_from_lancedb(
                start_date=current_start,
                end_date=current_end,
                category_filter=st.session_state.get("last_params", {}).get("main_cat", None),
                subcats=st.session_state.get("last_params", {}).get("subcats", None),
            )

        if current_papers:
            msg = f"Loaded {len(current_papers)} papers from LanceDB in this date range."
            if date_option == "All Time" and len(current_papers) >= DB_FETCH_LIMIT:
                msg += f" (capped at {DB_FETCH_LIMIT:,} most-recent — full corpus exceeds memory limits)"
            st.success(msg)
        else:
            st.info("No papers found in LanceDB for this date range.")

        # P-09: Early venue filter REMOVED — was double-filtering and blinding Stage 1.
        # Venue filter now runs correctly post-Stage 3 only (see below, after candidates are selected).
        # Previously this block applied filter_papers_by_venue immediately after fetch,
        # reducing Stage 1 input pool with no recall recovery path.

        st.session_state["current_papers"] = current_papers
    else:
        current_papers = st.session_state["current_papers"]

    if not current_papers:
        st.warning("No papers found for this date/venue combination. Please adjust filters.")
        return

    # Apply NOT filter provider-agnostically
    if not_text:
        current_papers, removed_count = filter_papers_by_not_terms(current_papers, not_text)
        st.info(f"Excluded {removed_count} papers whose title or abstract contained NOT terms (lexical).")
        # ponytail: Semantic NOT second pass retired — ran pre-Stage1, before paper_vectors
        # exist (LanceDB vectors are only fetched in _lancedb_hybrid_stage1). Re-add if
        # paraphrase/synonym exclusion matters enough to fetch vectors here too.

    st.session_state["current_papers"] = current_papers

    save_json(
        os.path.join(project_folder, "current_papers_all.json"),
        [asdict(p) for p in current_papers],
    )

    # 3. Candidate selection
    if mode == "global":
        st.subheader("3. Candidate Selection (Most Recent Papers)")
        if run_clicked or "candidates" not in st.session_state:
            sorted_papers = sorted(
                current_papers,
                key=lambda p: p.submitted_date,
                reverse=True,
            )
            candidates = sorted_papers[:150] if len(sorted_papers) > 150 else sorted_papers
            candidates = enrich_paper_signals(candidates)
            st.session_state["candidates"] = candidates
        else:
            candidates = st.session_state["candidates"]

        st.success(
            f"{len(candidates)} most recent cs.AI, cs.LG, and cs.HC papers selected as candidates (global mode)."
        )
    else:
        st.subheader("3. Embedding Based Candidate Selection")
        if run_clicked or "candidates" not in st.session_state:
            _acquired = _PIPELINE_SEMAPHORE.acquire(timeout=90)
            if not _acquired:
                st.error("⚠️ Server is busy with other requests. Please retry in 30 seconds.")
                st.stop()
            try:
                with st.spinner("Selecting top candidate papers via embeddings..."):
                    candidates = select_embedding_candidates(
                        current_papers,
                        query_brief=query_brief,
                        llm_config=active_llm_config,
                        max_candidates=150,
                        qil_cache=st.session_state.setdefault("_qil_cache", {}),
                    )
            finally:
                _PIPELINE_SEMAPHORE.release()

            if not candidates:
                st.warning("Embedding stage returned no candidates. Using all fetched papers as fallback.")
                candidates = current_papers
            st.session_state["candidates"] = candidates
        else:
            candidates = st.session_state["candidates"]

        st.success(f"{len(candidates)} top candidates selected by embedding similarity for further filtering.")

    save_json(
        os.path.join(project_folder, "candidates_embedding_selected.json"),
        [asdict(p) for p in candidates],
    )

    # P-09: Venue filter — applied ONCE, post-Stage 3, for all code paths
    if venue_filter_type != "None" and mode != "global":
        before_v = len(candidates)
        candidates = filter_papers_by_venue(
            candidates,
            venue_filter_type,
            selected_category,
            selected_venues,
        )
        after_v = len(candidates)
        display_sel = ", ".join(selected_venues) if selected_venues else ""
        name_string = f" → {display_sel}" if display_sel else ""
        st.info(
            f"Venue filter `{venue_filter_type}` applied{name_string} after Stage 3. "
            f"Remaining: {after_v} (filtered out {before_v - after_v})"
        )
        if not candidates:
            st.warning("Venue filter removed all candidates. Relaxing venue filter — showing all Stage 3 results.")
            candidates = st.session_state.get("candidates", current_papers)

    # 4. Relevance classification
    st.subheader("4. Relevance Classification")

    if mode == "global":
        st.info(
            "Global mode: no specific research brief was provided. "
            "Skipping relevance classification and treating all candidate papers as PRIMARY."
        )
        if run_clicked or any(p.focus_label is None for p in st.session_state.get("candidates", [])):
            for p in candidates:
                p.focus_label = "primary"
                p.llm_relevance_score = None
                if p.semantic_reason is None:
                    p.semantic_reason = "Global mode: no topical filtering; treated as primary."
    else:
        # Task 41: CrossEncoder Thresholding universally saves API tokens
        if run_clicked or any(p.focus_label is None for p in candidates):
            with st.spinner("Classifying candidates as PRIMARY, SECONDARY, or OFF TOPIC (CrossEncoder heuristic)..."):
                candidates = scibert_classify_papers(candidates)
            st.session_state["candidates"] = candidates
            st.success("Classifying candidates as PRIMARY, SECONDARY, or OFF TOPIC (CrossEncoder)... Done!")

    save_json(
        os.path.join(project_folder, "candidates_with_classification.json"),
        [asdict(p) for p in candidates],
    )

    # 5. Build prediction set with minimum size
    st.subheader("5. Automatically Selected Papers for Citation Impact Scoring")

    if mode == "global":
        primary_papers = [p for p in candidates]
        secondary_papers: List[Paper] = []
        used_papers = primary_papers.copy()
        used_label = "Global mode: all candidate papers treated as PRIMARY and used for citation impact scoring."
        st.success(
            f"Global mode: using {len(used_papers)} most recent cs.AI, cs.LG, and cs.HC papers for citation impact scoring."
        )
    else:
        primary_papers = [p for p in candidates if p.focus_label == "primary"]
        secondary_papers = [p for p in candidates if p.focus_label == "secondary"]

        for group in (primary_papers, secondary_papers):
            group.sort(
                key=lambda p: (
                    p.llm_relevance_score if p.llm_relevance_score is not None else 0.0,
                    p.semantic_relevance if p.semantic_relevance is not None else 0.0,
                ),
                reverse=True,
            )

        used_label = ""
        if primary_papers:
            if len(primary_papers) >= MIN_FOR_PREDICTION:
                used_papers = primary_papers.copy()
                used_label = "All PRIMARY papers (enough for citation impact scoring)"
                st.success(
                    f"{len(primary_papers)} papers classified as PRIMARY. "
                    f"Using all of them for citation impact scoring (≥ {MIN_FOR_PREDICTION})."
                )
            else:
                used_papers = primary_papers.copy()
                if secondary_papers:
                    needed = MIN_FOR_PREDICTION - len(primary_papers)
                    topups = secondary_papers[:needed]
                    used_papers.extend(topups)
                    total = len(used_papers)
                    if len(secondary_papers) >= needed:
                        used_label = f"PRIMARY + top {len(topups)} SECONDARY to reach about {MIN_FOR_PREDICTION}"
                        st.success(
                            f"{len(primary_papers)} papers classified as PRIMARY. "
                            f"Added {len(topups)} top SECONDARY papers for citation impact scoring."
                        )
                    else:
                        used_label = (
                            f"All PRIMARY + all available SECONDARY "
                            f"(only {len(secondary_papers)} secondary papers, total {total} < {MIN_FOR_PREDICTION})"
                        )
                        st.info(
                            f"{len(primary_papers)} papers classified as PRIMARY. "
                            f"Only {len(secondary_papers)} SECONDARY papers available, so you have "
                            f"{total} papers in the scoring set (below the target of {MIN_FOR_PREDICTION})."
                        )
                else:
                    used_label = "All PRIMARY papers (no SECONDARY available)"
                    st.warning(
                        f"Only {len(primary_papers)} PRIMARY papers and no SECONDARY. "
                        "Using all PRIMARY papers for citation impact scoring even though this is below the "
                        f"target of {MIN_FOR_PREDICTION}."
                    )
        elif secondary_papers:
            used_papers = secondary_papers[:20]
            used_label = f"Top 20 SECONDARY papers (no PRIMARY matches found)"
            st.warning(
                f"No papers were classified as PRIMARY. Using the top 20 SECONDARY matches "
                f"(out of {len(secondary_papers)} available). These may only partially match your brief."
            )
        else:
            st.error("No candidates were classified as PRIMARY or SECONDARY. Nothing to proceed with.")
            return

    used_papers.sort(
        key=lambda p: (
            p.llm_relevance_score if p.llm_relevance_score is not None else 0.0,
            p.semantic_relevance if p.semantic_relevance is not None else 0.0,
        ),
        reverse=True,
    )

    st.session_state["used_papers"] = used_papers
    st.session_state["used_label"] = used_label

    save_json(
        os.path.join(project_folder, "used_papers_for_prediction.json"),
        [asdict(p) for p in used_papers],
    )

    st.write(
        "These are the papers that the pipeline will use for citation impact scoring. "
        "Selection is automatic based on mode, embeddings (in targeted modes), and relevance classification."
    )
    st.write(f"**Citation impact set description:** {used_label}")
    st.write(f"**Number of papers in citation impact set:** {len(used_papers)}")

    for p in used_papers:
        with st.expander(p.title, expanded=False):
            # Task 38 Badges
            badges = []
            if p.has_code:
                badges.append("💻 **Code**")
            if p.has_dataset:
                badges.append("📊 **Dataset**")
            if p.paper_type_tag:
                badges.append(f"🏷️ **{p.paper_type_tag}**")
            if badges:
                st.markdown(" ".join(badges))
                
            st.write(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
            st.write(f"**Submitted:** {p.submitted_date.date().isoformat()}")
            st.write(f"[arXiv link]({p.arxiv_url}) | [PDF link]({p.pdf_url})")
            if p.focus_label:
                st.write(f"**Focus label:** {p.focus_label}")
            # P-10: show CE score (Stage 3) and cosine (Stage 2) as distinct signals
            if p.cross_encoder_score is not None:
                st.write(f"**CrossEncoder score (Stage 3):** {p.cross_encoder_score:.3f}")
            else:
                rel_str = f"{p.llm_relevance_score:.2f}" if p.llm_relevance_score is not None else "N/A"
                st.write(f"**Relevance score (heuristic):** {rel_str}")
            sim_str = f"{p.semantic_relevance:.3f}" if p.semantic_relevance is not None else "N/A"
            st.write(f"**Semantic similarity (Stage 2):** {sim_str}")
            if p.semantic_reason:
                st.write("**Why this paper matches your brief:**")
                st.write(p.semantic_reason)
            # ── Task 37: Retrieval provenance (visible when Task 33 RRF is active) ──
            if p.retrieval_source is not None:
                src_label = {
                    "both": "🔵 Full-text + vector (both)",
                    "bm25_only": "🟠 Full-text only",
                    "faiss_only": "🟣 Vector only",
                }.get(p.retrieval_source, p.retrieval_source)
                rrf_str = f"{p.rrf_score:.4f}" if p.rrf_score is not None else "N/A"
                bm25_str = f"#{p.bm25_rank}" if p.bm25_rank is not None else "—"
                faiss_str = f"#{p.faiss_rank}" if p.faiss_rank is not None else "—"
                st.caption(
                    f"📡 Retrieval: {src_label} · RRF score: {rrf_str} · "
                    f"Full-text rank: {bm25_str} · Vector rank: {faiss_str}"
                )
            st.write("**Abstract:**")
            st.write(p.abstract)

    selected_papers = used_papers
    save_json(
        os.path.join(project_folder, "selected_papers_for_prediction.json"),
        [asdict(p) for p in selected_papers],
    )

    # 6. Citation impact scoring
    st.subheader("6. Citation Impact Scoring")

    if provider in ("openai", "gemini", "groq"):
        st.markdown("""
**How this step works (Moneyball Edition)**

We use the **Moneyball Algorithm** to predict 1-year citation impact scores.
It combines four signals:
1. **Author Fame:** Query Semantic Scholar for author citations.
2. **Content Utility:** LLM rates abstract market fit.
3. **Hype Keywords:** Bonus for trending topics.
4. **Niche Penalties:** Penalty for small fields.

**Note on New Papers:** Papers less than 5 days old often lack citation data in Semantic Scholar. These are marked as **"Too new for impact score"** and ranked purely by their relevance to your query.
""")
    else:
        st.markdown("""
**How this step works (free local mode)**

In free local mode, the agent does not call any external LLM. Instead, it combines the embedding based similarity and relevance scores into a single numeric citation impact score and uses that score as a proxy for how influential the paper might be relative to others in this batch. The absolute numbers are less important than the relative ranking.

These scores are heuristic and should be used as a guide for exploration rather than as formal evaluation metrics.
        """)

    if run_clicked or "ranked_papers" not in st.session_state:
        if provider in ("openai", "gemini", "groq"):
            # Recover this query's QIL quality_modifier from the cache select_embedding_candidates
            # populated (same md5 key derivation) so Moneyball weighting matches Stage 1's pre-filter.
            _qil_key = hashlib.md5(query_brief.strip().lower().encode()).hexdigest()
            _cached_sq = st.session_state.get("_qil_cache", {}).get(_qil_key)
            quality_modifier = _cached_sq.quality_modifier if _cached_sq else "any"

            with st.spinner("Calling LLM API to compute citation impact scores for selected papers..."):
                papers_with_pred = predict_citations_direct(
                    target_papers=selected_papers,
                    llm_config=active_llm_config,
                    quality_modifier=quality_modifier,
                )
        else:
            with st.spinner("Computing heuristic citation impact scores from relevance signals..."):
                papers_with_pred = assign_heuristic_citations_free(selected_papers)

        # SEPARATE into groups by focus (Primary vs Secondary vs Others)
        primaries = [p for p in papers_with_pred if p.focus_label == "primary"]
        secondaries = [p for p in papers_with_pred if p.focus_label == "secondary"]
        others = [p for p in papers_with_pred if p.focus_label not in ("primary", "secondary")]

        # Define a helper to sort any group: Scored (High->Low) THEN Unscored (High Relevance->Low)
        def sort_group(group: List[Paper]) -> List[Paper]:
            scored = [p for p in group if p.predicted_citations is not None and p.predicted_citations >= 0]
            unscored = [p for p in group if p.predicted_citations == -1.0]
            
            # Sort Scored by predicted_citations desc
            scored.sort(key=lambda p: p.predicted_citations, reverse=True)
            
            # Sort Unscored by relevance desc
            unscored.sort(key=lambda p: (
                p.llm_relevance_score if p.llm_relevance_score is not None else 0.0,
                p.semantic_relevance if p.semantic_relevance is not None else 0.0
            ), reverse=True)
            
            return scored + unscored

        ranked_papers = sort_group(primaries) + sort_group(secondaries) + sort_group(others)
        
        st.session_state["ranked_papers"] = ranked_papers
        st.session_state["has_run_once"] = True
        # === Release transient memory after pipeline completes ===
        import gc
        gc.collect()
    else:
        ranked_papers = st.session_state["ranked_papers"]

    save_json(
        os.path.join(project_folder, "selected_papers_with_predictions.json"),
        [asdict(p) for p in ranked_papers],
    )

    # 7. All selected papers ranked
    st.subheader("7. All Selected Papers (Ranked by Citation Impact Score)")

    st.caption(
        "Primary papers are ranked first (Scored → Too New), followed by Secondary papers (Scored → Too New)."
    )

    table_rows = []
    for rank, p in enumerate(ranked_papers, start=1):
        pred_val = p.predicted_citations
        if pred_val == -1.0:
            pred_display = "Too new to rate"
        else:
            pred_display = str(int(pred_val or 0)) # Force string to avoid mixed-type error
            
        focus = p.focus_label or "unknown"
        if focus == "primary":
            focus_display = "🟢 primary"
        elif focus == "secondary":
            focus_display = "🟡 secondary"
        elif focus == "off-topic":
            focus_display = "⚪ off-topic"
        else:
            focus_display = focus
        llm_rel = float(p.llm_relevance_score or 0.0)
        emb_rel = float(p.semantic_relevance or 0.0)
        table_rows.append(
            {
                "Rank": rank,
                "Citation impact score (1y)": pred_display,
                "Focus": focus_display,
                "Relevance score": round(llm_rel, 2),
                "Embedding similarity": round(emb_rel, 3),
                "Venue":p.venue or "N/A",
                "Title": p.title,
                "arXiv": p.arxiv_url,

            }
        )

    df = pd.DataFrame(table_rows)

    # --- CSV Export ---
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download ranked results (CSV)",
        data=csv_bytes,
        file_name=f"ranked_papers_{timestamp}.csv",
        mime="text/csv",
    )

    if not df.empty:
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "arXiv": st.column_config.LinkColumn(
                    label="arXiv",
                    help="Open arXiv page",
                    validate="^https?://.*",
                    max_chars=100,
                    display_text="arXiv link"
                ),
                "Citation impact score (1y)": st.column_config.TextColumn(
                    label="Citation impact score (1y)",
                    help="Score or 'Too new to rate'"
                )
            }
        )

    # 8. Top N highlighted
    top_n_effective = min(top_n, len(ranked_papers))
    topN = ranked_papers[:top_n_effective]
    st.session_state["topN"] = topN

    st.subheader(f"8. Top {top_n_effective} Papers (Highlighted)")

    if "plain_summaries" not in st.session_state:
        st.session_state["plain_summaries"] = {}
    plain_summaries: Dict[str, str] = st.session_state["plain_summaries"]

    for rank, p in enumerate(topN, start=1):
        st.markdown(f"### #{rank}: {p.title}")
        
        # Task 38 Badges
        badges = []
        if p.has_code:
            badges.append("💻 **Code**")
        if p.has_dataset:
            badges.append("📊 **Dataset**")
        if p.paper_type_tag:
            badges.append(f"🏷️ **{p.paper_type_tag}**")
        if p.reproducibility_score > 0:
            badges.append(f"🔄 **Reproducibility: {p.reproducibility_score}/3**")
        if badges:
            st.markdown(" ".join(badges))
            
        pred_val = p.predicted_citations
        if pred_val == -1.0:
            st.write(f"**Citation impact score (1 year):** Too new to rate")
        else:
            st.write(f"**Citation impact score (1 year):** {int(pred_val or 0)}")
            
        st.write(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
        st.markdown(f"**Venue:** {p.venue or 'N/A'}")
        st.write(f"[arXiv link]({p.arxiv_url}) | [PDF link]({p.pdf_url})")

        if provider in ("openai", "gemini", "groq"):
            paper_key = p.arxiv_id or p.title
            if paper_key in plain_summaries:
                summary = plain_summaries[paper_key]
            else:
                with st.spinner("Generating plain English summary..."):
                    summary = summarize_paper_plain_english(p, llm_config)
                plain_summaries[paper_key] = summary
                st.session_state["plain_summaries"] = plain_summaries

            st.markdown("**Plain English summary:**")
            st.write(summary)

            if p.prediction_explanations:
                st.write("**Why this citation impact score:**")
                for ex in p.prediction_explanations[:3]:
                    st.write(f"- {ex}")
                # REMOVED REDUNDANT CHECK HERE
                # The logic inside predict_citations_direct ALREADY adds the relevance insight
                # to prediction_explanations if score is -1.0.
                # So we just print the list (which we did above).

        else:
            st.markdown("**Plain English summary:** only available in OpenAI / Gemini / Groq options")
            st.markdown("**Why this citation impact score:** only available in OpenAI / Gemini / Groq options")

        if p.focus_label:
            st.write(f"**Focus label:** {p.focus_label}")
        # P-10: separate CE score and Stage 2 cosine for transparency
        if p.cross_encoder_score is not None:
            st.write(f"**CrossEncoder score (Stage 3):** {p.cross_encoder_score:.3f}")
        else:
            rel_str = f"{p.llm_relevance_score:.2f}" if p.llm_relevance_score is not None else "N/A"
            st.write(f"**Relevance score (heuristic):** {rel_str}")
        sim_str = f"{p.semantic_relevance:.3f}" if p.semantic_relevance is not None else "N/A"
        st.write(f"**Semantic similarity (Stage 2):** {sim_str}")

        if p.semantic_reason:
            st.write("**Why this paper matches your brief:**")
            st.write(p.semantic_reason)

        # ── Task 37: Retrieval provenance (visible when Task 33 RRF is active) ──
        if p.retrieval_source is not None:
            src_label = {
                "both": "🔵 Full-text + vector (both)",
                "bm25_only": "🟠 Full-text only",
                "faiss_only": "🟣 Vector only",
            }.get(p.retrieval_source, p.retrieval_source)
            rrf_str = f"{p.rrf_score:.4f}" if p.rrf_score is not None else "N/A"
            bm25_str = f"#{p.bm25_rank}" if p.bm25_rank is not None else "—"
            faiss_str = f"#{p.faiss_rank}" if p.faiss_rank is not None else "—"
            st.caption(
                f"📡 Retrieval: {src_label} · RRF score: {rrf_str} · "
                f"Full-text rank: {bm25_str} · Vector rank: {faiss_str}"
            )

        st.write("**Abstract:**")
        st.write(p.abstract)
        st.markdown("---")

    # 9. Markdown report for top N
    st.subheader("9. Export Top N Report")

    report_lines = [
        f"# Top {top_n_effective} Papers (Citation Impact Scores) - {datetime.now().isoformat()}",
        "## Research Brief",
        research_brief,
        "",
        "## Not Looking For (optional)",
        not_looking_for or "(none provided)",
        "",
        f"Mode: {mode}",
        f"Date range: {current_start} to {current_end}",
        f"Provider: {'OpenAI' if provider == 'openai' else 'Gemini' if provider == 'gemini' else 'Groq' if provider == 'groq' else 'Free local heuristic'}",
        f"Chat model: {model_name}",
        f"Embedding model: {embedding_model_name}",
        "",
    ]
    for rank, p in enumerate(topN, start=1):
        report_lines.append(f"## #{rank}: {p.title}")
        
        pred_val = p.predicted_citations
        if pred_val == -1.0:
            report_lines.append(f"- Citation impact score (1 year): Too new to rate")
        else:
            report_lines.append(f"- Citation impact score (1 year): {int(pred_val or 0)}")
            
        report_lines.append(f"- Authors: {', '.join(p.authors) if p.authors else 'Unknown'}")
        report_lines.append(f"- arXiv: {p.arxiv_url}")
        report_lines.append(f"- PDF: {p.pdf_url}")
        if p.focus_label:
            report_lines.append(f"- Focus label: {p.focus_label}")
        if p.llm_relevance_score is not None:
            report_lines.append(f"- Relevance score: {p.llm_relevance_score:.2f}")
        if p.semantic_relevance is not None:
            report_lines.append(f"- Embedding similarity: {p.semantic_relevance:.3f}")
        if p.semantic_reason:
            report_lines.append(f"- Relevance explanation: {p.semantic_reason}")
        if provider in ("openai", "gemini", "groq"):
            report_lines.append("- Citation impact explanations:")
            if p.prediction_explanations:
                for ex in p.prediction_explanations[:3]:
                    report_lines.append(f"  - {ex}")
                # Removed redundant check here too

        report_lines.append("")
        report_lines.append("Abstract:")
        report_lines.append(p.abstract)
        report_lines.append("")

    report_path = os.path.join(project_folder, "topN_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 10. ZIP download
    if run_clicked or "zip_bytes" not in st.session_state:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in [
                "config.json",
                "current_papers_all.json",
                "candidates_embedding_selected.json",
                "candidates_with_classification.json",
                "used_papers_for_prediction.json",
                "selected_papers_for_prediction.json",
                "selected_papers_with_predictions.json",
                "topN_report.md",
            ]:
                fpath = os.path.join(project_folder, fname)
                if os.path.exists(fpath):
                    zf.write(fpath, arcname=fname)
        zip_buffer.seek(0)
        st.session_state["zip_bytes"] = zip_buffer.getvalue()

    zip_bytes = st.session_state["zip_bytes"]

    st.success(f"Results saved in `{project_folder}`")
    st.write("- `current_papers_all.json`")
    st.write("- `candidates_embedding_selected.json`")
    st.write("- `candidates_with_classification.json`")
    st.write("- `used_papers_for_prediction.json`")
    st.write("- `selected_papers_for_prediction.json`")
    st.write("- `selected_papers_with_predictions.json`")
    st.write("- `topN_report.md`")

    st.download_button(
        "⬇️ Download all results as ZIP",
        data=zip_bytes,
        file_name=f"research_agent_{timestamp}.zip",
        mime="application/zip",
    )


if __name__ == "__main__":
    main()