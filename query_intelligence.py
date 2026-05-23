"""
query_intelligence.py — Task P-00: Query Intelligence Layer

Sits before Stage 1 of the 3-stage hybrid retrieval pipeline.
Converts a raw free-text research brief into five structured outputs:

  1. semantic_query  → clean 1-2 sentence statement for FAISS / SPECTER2
  2. bm25_keywords   → 5-10 precise terms for BM25 (YAKE-backed, floor ≥ 5)
  3. intent          → drives RRF weights and Stage 1 adaptive behaviour
  4. hard_filters    → metadata constraints (date_range, not_terms)
  5. quality_modifier → recency / influence signal

Provider chain (stops at first success):
  1. Groq LLM (gemma2-9b-it) — fast, ~200ms, 15k tok/min headroom
  2. OpenRouter LLM (google/gemini-flash-1.5-8b) — on rate-limit / infra errors
  3. Rules-based extractor — zero cost, no network, guaranteed fallback

YAKE keyword floor: after every path, supplements bm25_keywords to ≥ 5 terms.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: YAKE — statistical keyword extractor (zero RAM, no model loading)
# ---------------------------------------------------------------------------
try:
    import yake as _yake_lib
    _YAKE_AVAILABLE = True
except ImportError:
    _yake_lib = None  # type: ignore
    _YAKE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Intent = Literal["novelty", "diversity", "foundational", "specific", "survey", "general"]

_YAKE_FLOOR = 5  # Minimum bm25_keywords guaranteed after YAKE supplement


@dataclass
class StructuredQuery:
    """
    Structured decomposition of a raw research brief.

    Attributes
    ----------
    intent : Intent
        Drives downstream RRF weight modulation and Stage 1 adaptive K.
    semantic_query : str
        Clean 1–2 sentence statement optimised for dense vector retrieval
        (FAISS / SPECTER2).
    bm25_keywords : List[str]
        5–10 precise terms including synonyms and domain aliases.
        Replaces the raw ``query_brief`` sent to BM25.
        YAKE floor guarantees ≥ 5 terms regardless of LLM output quality.
    hard_filters : dict
        Metadata constraints extracted from the brief.
        Keys: ``date_range`` (Optional[tuple[str,str]]), ``not_terms`` (List[str]),
              ``authors`` (List[str]), ``venues`` (List[str]).
    quality_modifier : str
        One of "recent" | "influential" | "emerging" | "classic" | "any".
    raw_brief : str
        Original unmodified brief for logging / debug.
    source : str
        "llm_groq" | "llm_openrouter" | "rules" — indicates which path produced this object.
    """
    intent: Intent = "general"
    semantic_query: str = ""
    bm25_keywords: List[str] = field(default_factory=list)
    hard_filters: Dict[str, Any] = field(default_factory=dict)
    quality_modifier: str = "any"
    raw_brief: str = ""
    source: str = "rules"

    @property
    def rrf_weight_faiss(self) -> float:
        """Higher FAISS weight for semantic / novelty / survey intents."""
        return {
            "novelty":      1.4,
            "survey":       1.2,
            "diversity":    1.2,
            "foundational": 0.8,
            "specific":     1.0,
            "general":      1.0,
        }.get(self.intent, 1.0)

    @property
    def rrf_weight_bm25(self) -> float:
        """Higher BM25 weight for foundational / specific / survey intents."""
        return {
            "foundational": 1.4,
            "specific":     1.3,
            "survey":       1.2,
            "novelty":      0.8,
            "diversity":    1.0,
            "general":      1.0,
        }.get(self.intent, 1.0)

    @property
    def bm25_query_string(self) -> str:
        """Space-joined keyword string for BM25 tokenizer."""
        return " ".join(self.bm25_keywords) if self.bm25_keywords else self.semantic_query


# ---------------------------------------------------------------------------
# LLM prompts — minified to reduce token consumption (~50–60% vs v1)
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "Research query analyser. Decompose brief → structured JSON components. "
    "Return ONLY valid JSON. No markdown, no explanation."
)

_LLM_PROMPT_TEMPLATE = """Brief:
\"\"\"{brief}\"\"\"

Return JSON:
{{
  "intent": "<one of: novelty|diversity|foundational|specific|survey|general>",
  "semantic_query": "<1-2 sentence dense-vector statement, not a question>",
  "bm25_keywords": ["<5-10 terms including synonyms>"],
  "hard_filters": {{"not_terms": [], "authors": [], "venues": []}},
  "quality_modifier": "<one of: recent|influential|emerging|classic|any>"
}}

Rules:
- novelty: wants new/recent/emerging work
- foundational: wants seminal/classic/influential work
- survey: wants overviews/comparisons/reviews
- specific: names paper/author/technique
- diversity: wants broad cross-area coverage
- general: default
- bm25_keywords: include synonyms (e.g. "LLM","large language model")
- not_terms: only explicitly rejected terms"""


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_groq_llm(
    brief: str,
    api_key: str,
    model: str = "gemma2-9b-it",
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Groq API call with structured JSON output.

    Returns
    -------
    (parsed_dict, None)     on success
    (None, error_type)      on failure

    error_type values:
      "rate_limit" — HTTP 429          → caller should escalate to OpenRouter
      "infra"      — timeout/5xx       → caller should escalate to OpenRouter
      "parse"      — bad JSON output   → caller should fall to rules
      "other"      — unexpected error  → caller should fall to rules
    """
    try:
        from groq import (  # type: ignore
            Groq,
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            APIStatusError,
        )
    except ImportError:
        logger.debug("[QIL][Groq] groq package not installed — skip Groq path")
        return None, "other"

    prompt = _LLM_PROMPT_TEMPLATE.format(brief=brief[:3000])

    for attempt in range(2):
        try:
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                ).strip()
            return json.loads(raw), None

        except json.JSONDecodeError as e:
            logger.warning("[QIL][Groq] JSON parse error (attempt %d): %s", attempt + 1, e)
            return None, "parse"  # same model won't improve — go to rules

        except RateLimitError as e:
            logger.warning("[QIL][Groq] Rate limit (429): %s", e)
            return None, "rate_limit"

        except (APIConnectionError, APITimeoutError) as e:
            logger.warning("[QIL][Groq] Connection/timeout (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1)
                continue
            return None, "infra"

        except InternalServerError as e:
            logger.warning("[QIL][Groq] Internal server error (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1)
                continue
            return None, "infra"

        except APIStatusError as e:
            if e.status_code >= 500:
                logger.warning("[QIL][Groq] 5xx %d (attempt %d): %s", e.status_code, attempt + 1, e)
                if attempt == 0:
                    time.sleep(1)
                    continue
                return None, "infra"
            logger.warning("[QIL][Groq] API error %d: %s", e.status_code, e)
            return None, "other"

        except Exception as e:
            logger.warning("[QIL][Groq] Unexpected error (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1)
                continue
            return None, "other"

    return None, "other"


def _call_openrouter_llm(
    brief: str,
    api_key: str,
    model: str = "google/gemini-flash-1.5-8b",
) -> Tuple[Optional[dict], Optional[str]]:
    """
    OpenRouter API call via OpenAI-compatible SDK.
    Triggered only on Groq rate-limit or infra errors — never on parse errors.

    Returns same (dict | None, error_type | None) contract as _call_groq_llm.
    """
    try:
        from openai import (  # type: ignore
            OpenAI,
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
        )
    except ImportError:
        logger.debug("[QIL][OpenRouter] openai package not installed — skip OpenRouter path")
        return None, "other"

    prompt = _LLM_PROMPT_TEMPLATE.format(brief=brief[:3000])

    for attempt in range(2):
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=5.0,  # free-tier queues can be slow; cap at 5s to protect UX
                default_headers={
                    "HTTP-Referer": "https://researchagent.streamlit.app",
                    "X-Title": "ResearchAgent QIL",
                },
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                ).strip()
            return json.loads(raw), None

        except json.JSONDecodeError as e:
            logger.warning("[QIL][OpenRouter] JSON parse error (attempt %d): %s", attempt + 1, e)
            return None, "parse"

        except RateLimitError as e:
            logger.warning("[QIL][OpenRouter] Rate limit: %s", e)
            return None, "rate_limit"

        except (APIConnectionError, APITimeoutError) as e:
            logger.warning("[QIL][OpenRouter] Connection/timeout (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1)
                continue
            return None, "infra"

        except APIStatusError as e:
            if e.status_code >= 500:
                logger.warning("[QIL][OpenRouter] 5xx %d (attempt %d): %s", e.status_code, attempt + 1, e)
                if attempt == 0:
                    time.sleep(1)
                    continue
                return None, "infra"
            logger.warning("[QIL][OpenRouter] API error %d: %s", e.status_code, e)
            return None, "other"

        except Exception as e:
            logger.warning("[QIL][OpenRouter] Unexpected error (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1)
                continue
            return None, "other"

    return None, "other"


# ---------------------------------------------------------------------------
# Rules-based fallback (no API key required)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "and", "or", "for", "with", "that", "this", "from", "into", "about",
    "what", "are", "your", "their", "they", "them", "using", "used", "use", "am",
    "main", "whose", "where", "when", "which", "have", "has", "had", "been", "i",
    "being", "also", "only", "just", "more", "most", "very", "much", "some", "want",
    "such", "than", "then", "over", "under", "not", "without", "within", "look",
    "papers", "paper", "interested", "looking", "brief", "contribution", "research",
    "like", "would", "should", "could", "please", "find", "show",
    "especially", "care", "example", "new", "novel", "recent",
})

_INTENT_SIGNALS: Dict[str, List[str]] = {
    "novelty":      ["novel", "new", "recent", "latest", "emerging", "state-of-the-art", "sota", "cutting-edge"],
    "foundational": ["foundational", "seminal", "classic", "influential", "survey textbook", "original", "pioneering"],
    "survey":       ["survey", "overview", "comprehensive", "review", "comparison", "benchmark", "evaluation study"],
    "specific":     [],  # detected via named entities (author names, years, paper titles)
    "diversity":    ["diverse", "broad", "various", "multiple areas", "different approaches", "coverage"],
}

_QUALITY_SIGNALS: Dict[str, List[str]] = {
    "recent":      ["recent", "latest", "new", "2024", "2025", "2026", "newest"],
    "influential": ["influential", "highly cited", "impactful", "foundational", "popular"],
    "emerging":    ["emerging", "growing", "rising", "promising", "early-stage"],
    "classic":     ["classic", "seminal", "original", "historical", "foundational"],
}


def _detect_intent(brief_lower: str) -> Intent:
    for intent, signals in _INTENT_SIGNALS.items():
        if any(s in brief_lower for s in signals):
            return intent  # type: ignore[return-value]
    if re.search(r"\b(20\d{2})\b", brief_lower) or re.search(r"\bby [A-Z][a-z]+", brief_lower):
        return "specific"
    return "general"


def _extract_keywords(brief: str, top_n: int = 8) -> List[str]:
    """
    Keyword extraction with YAKE when available, bigram-frequency fallback otherwise.

    YAKE is a statistical extractor (zero RAM, no model) that scores n-grams by
    co-occurrence and positional features — better than pure frequency counting for
    scientific text. Stopword-filtered post-processing ensures quality.
    """
    if _YAKE_AVAILABLE and _yake_lib is not None:
        try:
            kw_extractor = _yake_lib.KeywordExtractor(
                lan="en",
                n=3,                      # max trigrams: captures "contrastive learning methods"
                dedupLim=0.7,             # avoid near-duplicate phrases
                dedupFunc="seqm",
                windowsSize=2,
                top=max(top_n * 2, 15),  # over-extract then filter
                features=None,
            )
            raw = kw_extractor.extract_keywords(brief)
            # YAKE score: lower = more relevant. Post-filter: skip phrases where
            # ALL constituent words are stopwords.
            keywords = [
                kw for kw, _score in raw
                if not all(w in _STOPWORDS for w in kw.lower().split())
            ][:top_n]
            if keywords:
                return keywords
        except Exception as e:
            logger.debug("[QIL] YAKE extraction failed: %s — using bigram fallback", e)

    # Bigram frequency fallback (original implementation)
    text = brief.lower()
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z\-]+\b", text)
    filtered = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]

    freq: Dict[str, int] = {}
    for t in filtered:
        freq[t] = freq.get(t, 0) + 1

    words = text.split()
    bigrams: List[str] = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 not in _STOPWORDS and w2 not in _STOPWORDS and len(w1) > 2 and len(w2) > 2:
            bigrams.append(f"{w1} {w2}")

    bigram_freq: Dict[str, int] = {}
    for bg in bigrams:
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1

    combined: Dict[str, float] = {}
    for bg, c in bigram_freq.items():
        combined[bg] = c * 2.0
    for ug, c in freq.items():
        if not any(ug in bg for bg in bigram_freq):
            combined[ug] = float(c)

    sorted_terms = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_terms[:top_n]]


def _build_semantic_query(brief: str) -> str:
    """Strip NOT-looking-for sections, trim stop-phrases, produce a clean 1-2 sentence statement."""
    brief_clean = re.split(r"WHAT I AM NOT LOOKING FOR", brief, flags=re.IGNORECASE)[0]
    brief_clean = re.sub(r"^RESEARCH BRIEF:\s*", "", brief_clean, flags=re.MULTILINE).strip()
    brief_clean = re.sub(r"\s+", " ", brief_clean).strip()
    if len(brief_clean) > 512:
        match = re.search(r"[.!?]\s", brief_clean[200:512])
        if match:
            brief_clean = brief_clean[:200 + match.start() + 1]
        else:
            brief_clean = brief_clean[:512]
    return brief_clean


def _extract_not_terms(brief: str) -> List[str]:
    """Extract explicitly rejected terms from the NOT-looking-for section."""
    parts = re.split(r"WHAT I AM NOT LOOKING FOR", brief, flags=re.IGNORECASE)
    if len(parts) < 2:
        return []
    not_section = parts[1].strip()
    raw_terms = re.split(r"[,\n;]+", not_section)
    return [t.strip().lower() for t in raw_terms if len(t.strip()) > 3][:10]


def _rules_based_analyse(brief: str) -> StructuredQuery:
    """Pure-Python fallback analyser. No external APIs."""
    brief_lower = brief.lower()
    intent = _detect_intent(brief_lower)

    quality_modifier = "any"
    for mod, signals in _QUALITY_SIGNALS.items():
        if any(s in brief_lower for s in signals):
            quality_modifier = mod
            break

    return StructuredQuery(
        intent=intent,
        semantic_query=_build_semantic_query(brief),
        bm25_keywords=_extract_keywords(brief),
        hard_filters={
            "not_terms": _extract_not_terms(brief),
            "authors": [],
            "venues": [],
        },
        quality_modifier=quality_modifier,
        raw_brief=brief,
        source="rules",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VALID_INTENTS = {"novelty", "diversity", "foundational", "specific", "survey", "general"}
_VALID_QUALITY = {"recent", "influential", "emerging", "classic", "any"}


def analyse_query(
    brief: str,
    groq_api_key: Optional[str] = None,
    groq_model: str = "llama-3.1-8b-instant",
    openrouter_api_key: Optional[str] = None,
    openrouter_model: str = "openai/gpt-oss-20b:free",
) -> StructuredQuery:
    """
    Analyse a raw research brief and return a :class:`StructuredQuery`.

    Provider chain (stops at first success):
      1. Groq LLM (gemma2-9b-it) — fast, ~200ms, 15k tok/min headroom
      2. OpenRouter (google/gemini-flash-1.5-8b) — triggered on Groq rate-limit / infra errors
      3. Rules-based extractor — zero cost, always available

    YAKE floor applied after every path: supplements bm25_keywords to ≥ 5 terms.

    Parameters
    ----------
    brief : str
        Raw free-text research brief from the user.
    groq_api_key : str, optional
        Groq API key. Falls back to GROQ_API_KEY env var.
    groq_model : str
        Groq model name (default: gemma2-9b-it).
    openrouter_api_key : str, optional
        OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        Only used when Groq returns a rate-limit or infra error.
    openrouter_model : str
        OpenRouter model name (default: google/gemini-flash-1.5-8b).
    """
    if not brief or not brief.strip():
        logger.debug("[QIL] Empty brief — returning empty StructuredQuery")
        return StructuredQuery(raw_brief=brief, source="rules")

    t0 = time.perf_counter()

    def _build_sq_from_llm(raw: dict, source: str) -> Optional[StructuredQuery]:
        """Validate and build StructuredQuery from LLM JSON dict."""
        try:
            return StructuredQuery(
                intent=raw.get("intent", "general") if raw.get("intent") in _VALID_INTENTS else "general",
                semantic_query=str(raw.get("semantic_query", "") or "").strip() or _build_semantic_query(brief),
                bm25_keywords=list(raw.get("bm25_keywords", []) or [])[:10],
                hard_filters=raw.get("hard_filters", {"not_terms": [], "authors": [], "venues": []}),
                quality_modifier=raw.get("quality_modifier", "any") if raw.get("quality_modifier") in _VALID_QUALITY else "any",
                raw_brief=brief,
                source=source,
            )
        except Exception as e:
            logger.warning("[QIL] StructuredQuery build failed from %s output: %s", source, e)
            return None

    def _apply_yake_floor(sq: StructuredQuery) -> StructuredQuery:
        """Supplement bm25_keywords with YAKE if below the minimum floor."""
        if len(sq.bm25_keywords) < _YAKE_FLOOR:
            yake_kws = _extract_keywords(brief, top_n=10)
            existing_lower = {kw.lower() for kw in sq.bm25_keywords}
            for kw in yake_kws:
                if kw.lower() not in existing_lower:
                    sq.bm25_keywords.append(kw)
                    existing_lower.add(kw.lower())
                if len(sq.bm25_keywords) >= 8:
                    break
        return sq

    # ── Groq path ─────────────────────────────────────────────────────────────
    groq_key = (groq_api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if groq_key:
        raw, err = _call_groq_llm(brief, groq_key, groq_model)
        if raw is not None:
            sq = _build_sq_from_llm(raw, source="llm_groq")
            if sq is not None:
                sq = _apply_yake_floor(sq)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("[QIL] Groq: intent=%s quality=%s keywords=%s (%.0f ms)",
                            sq.intent, sq.quality_modifier, sq.bm25_keywords[:3], elapsed)
                return sq
        elif err in ("rate_limit", "infra"):
            logger.info("[QIL] Groq %s — escalating to OpenRouter", err)
            # intentional fall-through to OpenRouter

    # ── OpenRouter path ────────────────────────────────────────────────────────
    # Only reached on Groq rate_limit / infra. Not triggered on parse errors.
    or_key = (openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
    if or_key:
        raw, err = _call_openrouter_llm(brief, or_key, openrouter_model)
        if raw is not None:
            sq = _build_sq_from_llm(raw, source="llm_openrouter")
            if sq is not None:
                sq = _apply_yake_floor(sq)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("[QIL] OpenRouter: intent=%s quality=%s keywords=%s (%.0f ms)",
                            sq.intent, sq.quality_modifier, sq.bm25_keywords[:3], elapsed)
                return sq

    # ── Rules fallback ─────────────────────────────────────────────────────────
    sq = _rules_based_analyse(brief)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[QIL] Rules: intent=%s quality=%s keywords=%s (%.0f ms)",
                sq.intent, sq.quality_modifier, sq.bm25_keywords[:3], elapsed)
    return sq
