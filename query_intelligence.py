"""
query_intelligence.py — Task P-00: Query Intelligence Layer

Sits before Stage 1 of the 3-stage hybrid retrieval pipeline.
Converts a raw free-text research brief into six structured outputs:

  1. semantic_query  → clean 1-2 sentence statement for SPECTER2 / LanceDB vector
  2. bm25_keywords   → 8-12 quality-filtered terms for LanceDB FTS (YAKE-backed)
  3. intent          → drives RRF weights and Stage 1 adaptive behaviour
  4. hard_filters    → metadata constraints (date_range, not_terms)
  5. quality_modifier → recency / influence signal
  6. criteria        → 0-5 relevance criteria, display-only (QIL v3 Stage 2) —
                        not yet wired into retrieval or ranking (Stage 3)

Provider chain (stops at first success):
  1. Groq LLM (qwen/qwen3.6-27b, reasoning disabled) — fast, ~900ms, 15k tok/min headroom
  2. OpenRouter LLM (google/gemma-4-26b-a4b-it:free) — on rate-limit / infra errors
  3. Rules-based extractor — zero cost, no network, guaranteed fallback

Both LLM legs request `response_format={"type": "json_object"}` and drop it on a
400 so a model that doesn't support JSON mode still gets one free-text attempt.

YAKE keyword floor: tops a thin LLM keyword list up to _YAKE_FLOOR terms.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
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

# Top-up target for bm25_keywords after any LLM path (A1/R8 raised it 5 -> 8, to
# match the wider synonym-expanded keyword set the prompt now asks for). Not an
# absolute guarantee: YAKE is statistical, so a six-word brief simply has fewer
# than 8 extractable terms and the result is short by definition.
_YAKE_FLOOR = 8

# Ceiling on what an LLM path may contribute (QIL v3 Stage 1: 15 -> 12 — term
# quality matters more than count, and a lower cap discourages the keyword-bloat
# that dilutes FTS with generic filler).
_MAX_LLM_KEYWORDS = 12

# Brief length (chars) below which semantic_query uses the brief's own text
# instead of an LLM paraphrase. See _apply_verbatim_query_floor.
_VERBATIM_QUERY_MAX_CHARS = 150


def _coerce_year(value: Any) -> Optional[int]:
    """First 4-digit 19xx/20xx in `value` (int, "2022", "2022-01-01", date) or None."""
    if value is None:
        return None
    match = re.search(r"(?:19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


# Author/venue names reach a LanceDB WHERE clause as LIKE literals. They come from
# free-text briefs, so this is a trust boundary: whitelist the characters real names
# use and drop everything else, rather than escaping quotes and hoping. A name that
# sanitises to empty is skipped, never partially applied.
_LIKE_SAFE_RE = re.compile(r"[^0-9A-Za-z .\-&]")
# Single hyphens are real ("Smith-Jones"); runs of them are the SQL comment marker.
# Quotes are already gone by here so a literal can't be terminated, but leaving `--`
# in place is the kind of thing that becomes a bug the day the quoting changes.
_LIKE_COLLAPSE_RE = re.compile(r"[-\s]{2,}")
_LIKE_MAX_LEN = 60


def _like_literal(value: str) -> str:
    """Strip a name down to a LIKE-safe literal. Returns '' when nothing survives."""
    cleaned = _LIKE_SAFE_RE.sub("", str(value))
    return _LIKE_COLLAPSE_RE.sub(" ", cleaned)[:_LIKE_MAX_LEN].strip()


def _coerce_str_list(value: Any) -> List[str]:
    """Normalise an LLM-supplied field to a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [s.strip() for s in (str(v) for v in value) if s.strip()]


@dataclass
class HardFilters:
    """
    Typed metadata constraints extracted from the brief (B5 / Gap 8).

    Replaces the former ``Dict[str, Any]``, whose untyped `.get("date_range")`
    access is the documented reason `date_range` was never wired downstream.

    Citation bounds are deliberately absent. `quality_modifier` already owns the
    `citation_count` clauses via ``QUALITY_LANCEDB_FILTERS`` in `pipeline_core`;
    a second producer for the same clause is how the original dict drifted.

    `authors`/`venues` are held but not emitted by :meth:`to_lancedb_filter` —
    B3/B4 apply them as ``LIKE`` filters with a result-count fallback, which
    can't be blindly AND-ed into an unconditional WHERE.
    """
    not_terms: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None

    @classmethod
    def from_raw(cls, raw: Any) -> "HardFilters":
        """
        Build from whatever the LLM returned. Never raises — an 8B model can
        emit a string where a list belongs, or omit the key entirely.

        Accepts `date_range` as ``{"from": ..., "to": ...}`` or as a
        ``(from, to)`` pair (the shape the original docstring specified).
        """
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            return cls()

        date_range = raw.get("date_range")
        if isinstance(date_range, (list, tuple)):
            raw_from = date_range[0] if len(date_range) > 0 else None
            raw_to = date_range[1] if len(date_range) > 1 else None
        elif isinstance(date_range, dict):
            raw_from, raw_to = date_range.get("from"), date_range.get("to")
        else:
            raw_from = raw_to = None

        return cls(
            not_terms=[t.lower() for t in _coerce_str_list(raw.get("not_terms"))],
            authors=_coerce_str_list(raw.get("authors")),
            venues=_coerce_str_list(raw.get("venues")),
            year_from=_coerce_year(raw_from),
            year_to=_coerce_year(raw_to),
        )

    def to_lancedb_filter(self) -> Optional[str]:
        """
        SQL-ish WHERE fragment for LanceDB's `.where()`, or None when unconstrained.

        Injection-safe by construction: `_coerce_year` yields ints or None, so no
        brief-supplied text reaches the clause string.
        """
        clauses = []
        if self.year_from is not None:
            clauses.append(f"year >= {self.year_from}")
        if self.year_to is not None:
            clauses.append(f"year <= {self.year_to}")
        return " AND ".join(clauses) if clauses else None

    def to_entity_filter(self) -> Optional[str]:
        """
        LIKE clause for brief-named authors/venues (B3/B4), or None.

        OR within a field, AND across fields: two venues means either venue, but an
        author *and* a venue means both. `authors` is a JSON-encoded list column, so
        this substring-matches the encoded text — good enough for name variants
        ("Geoffrey E. Hinton", "Hinton, G."), but a common surname can match inside
        unrelated encoded content. That is why the caller keeps a relax step.

        Applied separately from :meth:`to_lancedb_filter` because entity filters are
        relaxable and year bounds are not — dropping a year the user stated is a
        different kind of wrong from dropping an over-eager name match.
        """
        groups = []
        for column, values in (("authors", self.authors), ("venue", self.venues)):
            literals = [lit for lit in (_like_literal(v) for v in values) if lit]
            if literals:
                groups.append(" OR ".join(f"{column} LIKE '%{lit}%'" for lit in literals))
        return " AND ".join(f"({g})" for g in groups) if groups else None


@dataclass
class Criterion:
    """
    A single relevance criterion decomposed from the brief (QIL v3 Stage 2).

    Display-only in this stage — surfaced in "How your query was read" so
    criteria quality is observable before Stage 3 wires them into judging.
    Retrieval and ranking are unaffected by this field for now.

    Modelled on Asta.ai's observed behaviour and Ai2's own documented Paper
    Finder mechanism (a query decomposed into sub-criteria, each verified
    separately per paper — see docs/PLAN.md). Deliberately NOT Ai2's
    weighted-summing-to-1.0 scheme: a small model reliably picking one of two
    string labels is a safer bet than reliably summing floats to exactly 1.0.
    """
    name: str
    definition: str
    strength: Literal["must", "should"] = "should"

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["Criterion"]:
        """
        None on anything unusable — a malformed criterion is dropped, not
        defaulted, since a criterion with an empty definition is worse than
        no criterion at all (it would just be a blank line in the UI).
        """
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip()
        definition = str(raw.get("definition") or "").strip()
        if not name or not definition:
            return None
        strength = raw.get("strength")
        return cls(name=name, definition=definition, strength=strength if strength in ("must", "should") else "should")


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
        Up to 15 terms — core terms plus synonyms, abbreviations and domain
        aliases. Joined into the LanceDB FTS query string in place of the raw
        ``query_brief``. The YAKE floor tops a thin LLM result up to
        ``_YAKE_FLOOR``; only the LLM paths can produce true synonyms, since
        YAKE is statistical and has no domain knowledge (Gap 2/Gap 3).
    hard_filters : HardFilters
        Typed metadata constraints extracted from the brief — see
        :class:`HardFilters`. Was an untyped dict before B5.
    quality_modifier : str
        One of "recent" | "influential" | "emerging" | "classic" | "any".
    criteria : List[Criterion]
        0-5 relevance criteria decomposed from the brief (QIL v3 Stage 2).
        Display-only — see :class:`Criterion`. Empty on the rules path, since
        criteria decomposition needs genuine language understanding a
        deterministic fallback can't provide; no criteria beats fake ones.
    raw_brief : str
        Original unmodified brief for logging / debug.
    source : str
        "llm_groq" | "llm_openrouter" | "rules" — indicates which path produced this object.
    """
    intent: Intent = "general"
    semantic_query: str = ""
    bm25_keywords: List[str] = field(default_factory=list)
    hard_filters: HardFilters = field(default_factory=HardFilters)
    quality_modifier: str = "any"
    criteria: List[Criterion] = field(default_factory=list)
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
  "bm25_keywords": ["<8-12 terms: specific technical terms, not generic ML words>"],
  "hard_filters": {{"not_terms": [], "authors": [], "venues": [],
                    "date_range": {{"from": null, "to": null}}}},
  "quality_modifier": "<one of: recent|influential|emerging|classic|any>",
  "criteria": [{{"name": "<short label>", "definition": "<one testable sentence>",
                "strength": "<must|should>"}}]
}}

Rules:
- novelty: wants new/recent/emerging work
- foundational: wants seminal/classic/influential work
- survey: wants overviews/comparisons/reviews
- specific: names paper/author/technique
- diversity: wants broad cross-area coverage
- general: default
- bm25_keywords: expand only what the brief actually names. For an acronym,
  write out its own full name (not a different technique's) plus 2-3 close
  synonyms. For a general topic with no named technique, keep the keywords at
  the SAME level of generality as the brief — do not invent specific named
  sub-techniques, algorithms, or adjacent subfields the brief itself doesn't
  mention, even ones commonly discussed in that area. If the brief lists
  several possible types of contribution, reflect that same breadth — don't
  narrow to one specific method the brief didn't name.
  Do not include generic words like "model", "method", "approach", "novel",
  "recent", "artificial intelligence", or "machine learning" — every AI paper
  could claim those, so they add no signal. Target 8-12 specific terms.
  Terms only, no phrases.
- not_terms: only explicitly rejected terms
- authors/venues: only names the brief actually asks for
- date_range: years as integers. Explicit years are re-extracted deterministically,
  so use this for phrasing a regex can't read (e.g. "post-ChatGPT era" -> 2023)
- criteria: 2-5 distinct, independently-checkable requirements decomposed from the
  brief. Each is a testable statement about what a relevant paper must contain —
  not a restatement of the whole topic, and not one blob covering everything.
  "must" for a requirement the brief states or clearly implies as non-negotiable;
  "should" for a softer preference. Do not repeat what hard_filters/not_terms/
  quality_modifier already cover."""


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_groq_llm(
    brief: str,
    api_key: str,
    model: str = "qwen/qwen3.6-27b",
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
    use_json_mode = True  # dropped on a 400 — see APIStatusError handler below

    for attempt in range(2):
        try:
            client = Groq(api_key=api_key)
            extra_kwargs = {}
            if use_json_mode:
                extra_kwargs["response_format"] = {"type": "json_object"}
            if model.startswith("qwen/"):
                # qwen3.6 emits a <think> block straight into `content` and burns
                # the whole max_tokens budget on it unless reasoning is disabled —
                # confirmed live: finish_reason="length" at max_tokens=2000, still
                # mid-thought, zero JSON. Other Groq models reject this param outright
                # (400 "not supported with this model"), so it's gated to qwen only.
                extra_kwargs["reasoning_effort"] = "none"
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=900,  # 600 (R8 keyword expansion) + ~300 for 2-5 criteria (v3 Stage 2)
                **extra_kwargs,
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
            if e.status_code == 400 and use_json_mode:
                logger.warning("[QIL][Groq] 400 with JSON mode — retrying without it: %s", e)
                use_json_mode = False
                continue
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
    model: str = "google/gemma-4-26b-a4b-it:free",
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
    use_json_mode = True  # dropped on a 400 — see APIStatusError handler below

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
                max_tokens=900,  # 600 (R8 keyword expansion) + ~300 for 2-5 criteria (v3 Stage 2)
                **({"response_format": {"type": "json_object"}} if use_json_mode else {}),
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
            if e.status_code == 400 and use_json_mode:
                logger.warning("[QIL][OpenRouter] 400 with JSON mode — retrying without it: %s", e)
                use_json_mode = False
                continue
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

# Checked in this order in _detect_intent — first match wins, so the narrower
# categories go before novelty's broader one. "new"/"recent" are deliberately
# excluded from novelty: bare enough to false-positive on substrings ("renewable",
# "newton") even with word boundaries would still be too generic, and real recency
# is already quality_modifier's job, not intent's.
_INTENT_SIGNALS: Dict[str, List[str]] = {
    "foundational": ["foundational", "seminal", "classic", "influential", "survey textbook", "original", "pioneering"],
    "survey":       ["survey", "overview", "comprehensive", "review", "comparison", "benchmark", "evaluation study"],
    "diversity":    ["diverse", "broad", "various", "multiple areas", "different approaches", "coverage"],
    "novelty":      ["novel", "latest", "emerging", "state-of-the-art", "sota", "cutting-edge"],
    "specific":     [],  # detected via named entities (author names, years, paper titles)
}

# Word-boundary-anchored per intent, so "peer-reviewed" doesn't match "review" and
# "renewable"/"newton" can't match a bare signal word (Gap: _detect_intent previously
# did unbounded substring matching via `s in brief_lower`).
_INTENT_SIGNAL_PATTERNS: Dict[str, re.Pattern] = {
    # Trailing "s?" so plurals still match ("surveys", "benchmarks") — a bare \b
    # boundary alone would exclude them, since "survey" has no boundary before the
    # "s" in "surveys". Deliberately not a full pluralization engine (misses
    # "studies" for "study") — this covers the simple-plural case actually needed.
    intent: re.compile(r"\b(?:" + "|".join(re.escape(s) for s in signals) + r")s?\b", re.IGNORECASE)
    for intent, signals in _INTENT_SIGNALS.items()
    if signals
}

_QUALITY_SIGNALS: Dict[str, List[str]] = {
    "recent":      ["recent", "latest", "new", "2024", "2025", "2026", "newest"],
    "influential": ["influential", "highly cited", "impactful", "foundational", "popular"],
    "emerging":    ["emerging", "growing", "rising", "promising", "early-stage"],
    "classic":     ["classic", "seminal", "original", "historical", "foundational"],
}


def _detect_intent(brief: str) -> Intent:
    """
    Takes the raw brief (not pre-lowered) — matching is case-insensitive internally,
    and _extract_authors needs original casing preserved.

    Scans with exclusion clauses stripped (_strip_negation) — otherwise a rejected
    topic can flip the classification. Observed live: "...exclude surveys" matched
    the "survey" signal word inside the very clause rejecting it, and returned
    intent="survey" for a query explicitly asking to NOT get surveys.
    """
    positive = _strip_negation(brief)
    for intent, pattern in _INTENT_SIGNAL_PATTERNS.items():
        if pattern.search(positive):
            return intent  # type: ignore[return-value]
    if re.search(r"\b(20\d{2})\b", positive) or _extract_authors(positive):
        return "specific"
    return "general"


# Intent/quality words that leak into bm25_keywords as filler regardless of
# whether the user actually asked for recency (QIL v3 Stage 1, Gap 4 from the
# original audit). These are already captured properly by `intent` and
# `quality_modifier` — duplicating them as content keywords adds zero
# discriminative power to FTS (nearly every abstract can be framed as "novel").
_INTENT_LEAK_TERMS = frozenset({
    "novel", "new", "recent", "latest", "emerging", "state-of-the-art", "sota", "cutting-edge",
})

# Generic/hypernym content words (QIL v3 Stage 1). DF-informed from a 15,000-doc
# random sample of this corpus (docs/PLAN.md corpus probe, 2026-07-28) but NOT a
# mechanically-derived threshold cutoff — pure document-frequency has a real
# overlap problem in the middle range (e.g. "adversarial" and "sampling" tested
# LOWER-DF than "transformer" despite being more generic, because a generic
# concept's frequency often splits across several near-synonyms — "method",
# "approach", "technique", "algorithm" — none of which alone looks dominant).
# This list is DF-informed, hand-reviewed, not automatically thresholded.
# Whole-term match only: "model" is dropped, "diffusion model" survives.
_HYPERNYM_TERMS = frozenset({
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "neural networks",
    "model", "models", "method", "methods", "approach", "approaches",
    "technique", "techniques", "algorithm", "algorithms",
    "framework", "frameworks", "system", "systems",
    "research", "paper", "papers", "work", "data", "result", "results",
    "performance", "analysis", "learning", "training", "evaluation", "optimization",
    "generation", "text",
})

_TERM_QUALITY_BLACKLIST = _INTENT_LEAK_TERMS | _HYPERNYM_TERMS


def _is_acceptable_keyword(kw: str) -> bool:
    """
    Shared acceptance check for bm25_keywords, used by both the term-quality
    floor and the YAKE top-up. A hypernym/intent-leak term or a malformed
    prompt-echo artifact (contains ":", or starts with "brief" — the model
    echoing the prompt's own Brief-then-quoted-text wrapper) must be rejected
    regardless of which path produced the candidate.

    Previously duplicated: the term-quality floor had this check, but the
    YAKE top-up loop only checked the blacklist, not the artifact patterns —
    observed live, reproducibly, a keyword "recommendation systems:" survived
    because it arrived via YAKE's own top-up, after the term-quality floor
    had already run once.
    """
    normalized = kw.strip().lower()
    return (
        normalized not in _TERM_QUALITY_BLACKLIST
        and ":" not in kw
        and not normalized.startswith("brief")
    )


def _extract_keywords(brief: str, top_n: int = 8) -> List[str]:
    """
    Keyword extraction with YAKE when available, bigram-frequency fallback otherwise.

    YAKE is a statistical extractor (zero RAM, no model) that scores n-grams by
    co-occurrence and positional features — better than pure frequency counting for
    scientific text. Stopword-filtered post-processing ensures quality.

    Cleans via _build_semantic_query first — the real caller (backend/pipeline_core.py's
    build_query_brief) wraps every brief as "RESEARCH BRIEF:\\n{text}", and n-gram
    extraction is sensitive to word ADJACENCY across that artificial boundary in a way
    plain substring/keyword matching elsewhere in this module isn't. Observed live,
    reproducibly: bigram extraction on the raw wrapped text produced "brief: diffusion"
    as a keyword — the tail of "BRIEF:" paired with the first real word after it. Also
    drops the "WHAT I AM NOT LOOKING FOR" section, so a rejected topic can't contribute
    search keywords either.
    """
    brief = _build_semantic_query(brief)
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

    # Strip leading/trailing punctuation per word (keeps internal hyphens, so
    # "self-supervised" survives intact) before bigram-joining. Without this,
    # a word immediately followed by user punctuation — "systems:", "methods,"
    # — carries that punctuation into the joined bigram unfiltered, since a
    # stopword comparison against the punctuated form never matches. Observed
    # live, twice, with two different punctuation marks (":" then ",") on the
    # same brief — the same root cause, not two separate bugs.
    words = [re.sub(r"^[^a-zA-Z]+|[^a-zA-Z\-]+$", "", w) for w in text.split()]
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


_YEAR_PAT = r"(?:19|20)\d{2}"

# Ordered — first match wins per bound. Explicit range before open-ended bounds so
# "from 2020 to 2023" isn't read as an unbounded "from 2020".
_DATE_RANGE_RE = re.compile(
    rf"\b(?:from|between)?\s*({_YEAR_PAT})\s*(?:to|through|until|and|[-–—])\s*({_YEAR_PAT})\b",
    re.IGNORECASE,
)
_DATE_FROM_RE = re.compile(rf"\b(?:since|after|from|post-?)\s*({_YEAR_PAT})\b", re.IGNORECASE)
_DATE_TO_RE = re.compile(
    rf"\b(?:before|prior to|up to|until|earlier than|pre-?)\s*({_YEAR_PAT})\b", re.IGNORECASE
)
_DATE_LAST_N_RE = re.compile(r"\b(?:last|past)\s+(\d{1,2})\s*years?\b", re.IGNORECASE)

# Does the brief mention time *at all*? Gate for trusting an LLM-supplied
# date_range: without this an 8B model invents bounds for queries that never
# mentioned a period (observed: "diffusion models for text generation" -> 2020).
_TEMPORAL_HINT_RE = re.compile(
    r"(?:19|20)\d{2}|\b(?:recent|latest|current|modern|new|newest|last|past|since|"
    r"before|after|prior|upcoming|era|decade|year|month|old|classic|historical)\b",
    re.IGNORECASE,
)


def _extract_date_range(brief: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Year bounds stated explicitly in the brief (B2 / R6). ``(year_from, year_to)``.

    Deterministic patterns only — ranges, ``since YYYY``, ``before YYYY``,
    ``last N years``. A **bare** year is deliberately ignored: "the 2017
    Transformer paper" usually names a work rather than bounding the search, and
    a wrong year bound silently narrows the corpus to a single year. Natural
    language ("post-ChatGPT era") is left to the LLM's ``date_range``.
    """
    match = _DATE_RANGE_RE.search(brief)
    if match:
        first, second = int(match.group(1)), int(match.group(2))
        return (min(first, second), max(first, second))

    year_from = year_to = None

    last_n = _DATE_LAST_N_RE.search(brief)
    if last_n:
        year_from = datetime.now().year - int(last_n.group(1))

    if year_from is None:
        from_match = _DATE_FROM_RE.search(brief)
        if from_match:
            year_from = int(from_match.group(1))

    to_match = _DATE_TO_RE.search(brief)
    if to_match:
        year_to = int(to_match.group(1))

    return (year_from, year_to)


# Does the brief claim anything about impact? Gate for trusting an LLM-supplied
# `quality_modifier` of "influential", which pre-filters to citation_count >= 50.
_IMPACT_HINT_RE = re.compile(
    r"\b(?:influential|impact|impactful|cited|citation|seminal|landmark|"
    r"foundational|pioneering|important|significant|famous|popular|"
    r"well[- ]known|state[- ]of[- ]the[- ]art|sota)\b",
    re.IGNORECASE,
)

# "papers by Geoffrey Hinton and Yann LeCun" -> "Geoffrey Hinton and Yann LeCun".
# Only "by" — "from" is hopelessly ambiguous in briefs ("from 2020", "from NeurIPS").
_AUTHOR_BY_RE = re.compile(
    r"\b(?:authored\s+by|written\s+by|by)\s+"
    r"((?:[A-Z][A-Za-z.'\-]+)(?:\s+(?:and\s+|&\s+|,\s*)?[A-Z][A-Za-z.'\-]+)*)"
)
_AUTHOR_SPLIT_RE = re.compile(r"\s+and\s+|\s*&\s*|,\s*", re.IGNORECASE)
# Capitalised words that follow "by" without being a person.
_NOT_A_NAME = frozenset({
    "the", "a", "an", "using", "applying", "comparing", "combining", "leveraging",
    "improving", "training", "learning", "relevance", "date", "citation", "impact",
    "year", "default", "any", "all",
})
_MAX_AUTHORS = 5


def _extract_authors(brief: str) -> List[str]:
    """
    Author names stated with "by …" (B3 / R6-R7 follow-up).

    Deterministic counterpart to the LLM's `authors` field, which was observed
    returning `[]` for a brief that plainly said "by Hinton". Conservative: only
    the "by" construction, only capitalised tokens, blocklisted non-names.
    """
    names: List[str] = []
    seen = set()
    for match in _AUTHOR_BY_RE.finditer(brief):
        for candidate in _AUTHOR_SPLIT_RE.split(match.group(1)):
            name = candidate.strip()
            tokens = name.split()
            if not tokens or tokens[0].lower() in _NOT_A_NAME:
                continue
            if len(name) < 2 or name.lower() in seen:
                continue
            seen.add(name.lower())
            names.append(name)
            if len(names) >= _MAX_AUTHORS:
                return names
    return names


def _is_grounded_in(value: str, brief_lower: str) -> bool:
    """True when any distinctive token of `value` actually occurs in the brief.

    Guards against an LLM inventing an author or venue the user never mentioned —
    the filter would be a hard LIKE clause, so an invented one silently wrongs
    the whole result set.
    """
    tokens = [t for t in re.split(r"[^0-9A-Za-z]+", value) if len(t) >= 3]
    return any(t.lower() in brief_lower for t in tokens) if tokens else False


# Acronyms actually observed being mis-expanded by the LLM synonym-expansion feature
# (R8), or in the same confusable family — RL/alignment acronyms overlap across
# subfields more than vision/generative ones do. Deliberately NOT a comprehensive
# ML acronym list: an unlisted acronym is left untouched (Stage A can't verify it —
# Stage B's escalation call is separate future work), so a bloated table just adds
# maintenance surface with no benefit. Every entry's canonical value MUST pass
# _acronym_letters_match against its own key — see test_acronym_table_entries_pass_their_own_gate.
_ACRONYM_EXPANSIONS: Dict[str, str] = {
    "RLHF": "Reinforcement Learning from Human Feedback",
    "PPO": "Proximal Policy Optimization",
    "DPO": "Direct Preference Optimization",
    "KTO": "Kahneman-Tversky Optimization",
    "GRPO": "Group Relative Policy Optimization",
    "SFT": "Supervised Fine-Tuning",
}

_LETTER_GATE_STOPWORDS = frozenset({"of", "the", "from", "and", "for", "in", "on", "with", "a", "an"})


def _acronym_letters_match(acronym: str, phrase: str) -> bool:
    """
    True if `phrase`'s word-initial letters plausibly spell out `acronym`, in order.

    Two modes, either is accepted — no single mode covers every acronym style this
    corpus uses: "MoE" (Mixture of Experts) needs every word counted, while "RLHF"
    (Reinforcement Learning FROM Human Feedback) needs connector words skipped.
    Splits on all non-alphanumeric characters so hyphenated words count separately
    ("Kahneman-Tversky" -> "Kahneman", "Tversky", needed for KTO to match at all).

    This is the gate that keeps a merely-related keyword (RLHF -> "reward model")
    from ever being treated as a claimed expansion in the first place — "reward
    model"'s initials (R, M) don't correspond to RLHF's letters, so it's never even
    considered, let alone incorrectly "verified".
    """
    words = [w for w in re.split(r"[^0-9A-Za-z]+", phrase) if w]
    if not words:
        return False
    all_initials = "".join(w[0] for w in words).upper()
    skipped_initials = "".join(w[0] for w in words if w.lower() not in _LETTER_GATE_STOPWORDS).upper()
    target = acronym.upper()
    return target == all_initials or target == skipped_initials


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


_NOT_SECTION_RE = re.compile(r"WHAT I AM NOT LOOKING FOR", re.IGNORECASE)

# Inline exclusion phrasing. Captures to the end of the sentence/line only — a
# section header owns the rest of the brief, an inline phrase must not, or
# "exclude surveys. I also want X" would reject X too. These terms feed a
# substring filter over title+abstract (pipeline_core.py:1152), so over-capture
# silently deletes good papers; under-capture only misses an exclusion.
_NOT_INLINE_RE = re.compile(
    r"\b(?:exclude|excluding|omit|omitting|not including|not interested in|"
    r"no papers (?:on|about)|nothing (?:on|about))\b[:\s]*([^.\n;]+)",
    re.IGNORECASE,
)

# ponytail: split on and/or so "exclude surveys and benchmarks" yields two usable
# terms. Ceiling — it also splits genuine multi-word terms ("search and rescue").
# Upgrade path: only split when both sides survive the len>3 guard as noun phrases.
_NOT_TERM_SPLIT_RE = re.compile(r"[,\n;]+|\band\b|\bor\b", re.IGNORECASE)


def _strip_negation(brief: str) -> str:
    """
    Drop the NOT-looking-for section and inline exclusion clauses, keeping original
    casing (needed by _extract_authors). Used by _detect_intent so a rejected topic
    can't flip its own classification — see _detect_intent's docstring.
    """
    positive = _NOT_SECTION_RE.split(brief, maxsplit=1)[0]
    return _NOT_INLINE_RE.sub("", positive)


def _extract_not_terms(brief: str) -> List[str]:
    """
    Extract explicitly rejected terms from a brief.

    Two sources, both optional:
      * the ``WHAT I AM NOT LOOKING FOR`` section — owns everything after it
      * inline phrasing ("exclude X", "omit Y", "not interested in Z") in the
        text before that section — owns only to the end of its sentence
    """
    parts = _NOT_SECTION_RE.split(brief, maxsplit=1)
    head = parts[0]
    chunks = ([parts[1]] if len(parts) > 1 else []) + _NOT_INLINE_RE.findall(head)

    terms: List[str] = []
    seen = set()
    for chunk in chunks:
        for raw in _NOT_TERM_SPLIT_RE.split(chunk):
            term = raw.strip().strip("-–—:.").lower()
            if len(term) > 3 and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms[:10]


def _rules_based_analyse(brief: str) -> StructuredQuery:
    """Pure-Python fallback analyser. No external APIs."""
    brief_lower = brief.lower()
    intent = _detect_intent(brief)

    quality_modifier = "any"
    for mod, signals in _QUALITY_SIGNALS.items():
        if any(s in brief_lower for s in signals):
            quality_modifier = mod
            break

    year_from, year_to = _extract_date_range(brief)

    return StructuredQuery(
        intent=intent,
        semantic_query=_build_semantic_query(brief),
        bm25_keywords=_extract_keywords(brief),
        hard_filters=HardFilters(
            not_terms=_extract_not_terms(brief),
            authors=_extract_authors(brief),
            year_from=year_from,
            year_to=year_to,
        ),
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
    groq_model: str = "qwen/qwen3.6-27b",
    openrouter_api_key: Optional[str] = None,
    openrouter_model: str = "google/gemma-4-26b-a4b-it:free",
) -> StructuredQuery:
    """
    Analyse a raw research brief and return a :class:`StructuredQuery`.

    Provider chain (stops at first success):
      1. Groq LLM (qwen/qwen3.6-27b, reasoning disabled) — fast, ~900ms, 15k tok/min headroom
      2. OpenRouter (google/gemma-4-26b-a4b-it:free) — triggered on Groq rate-limit / infra errors
      3. Rules-based extractor — zero cost, always available

    YAKE floor applied after every LLM path: tops bm25_keywords up to _YAKE_FLOOR.

    Parameters
    ----------
    brief : str
        Raw free-text research brief from the user.
    groq_api_key : str, optional
        Groq API key. Falls back to GROQ_API_KEY env var.
    groq_model : str
        Groq model name (default: qwen/qwen3.6-27b). Reasoning is auto-disabled
        for any "qwen/"-prefixed model — see _call_groq_llm.
    openrouter_api_key : str, optional
        OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        Only used when Groq returns a rate-limit or infra error.
    openrouter_model : str
        OpenRouter model name (default: google/gemma-4-26b-a4b-it:free).
    """
    if not brief or not brief.strip():
        logger.debug("[QIL] Empty brief — returning empty StructuredQuery")
        return StructuredQuery(raw_brief=brief, source="rules")

    t0 = time.perf_counter()

    def _build_sq_from_llm(raw: dict, source: str) -> Optional[StructuredQuery]:
        """Validate and build StructuredQuery from LLM JSON dict."""
        try:
            criteria_raw = raw.get("criteria") or []
            criteria = [
                c for c in (Criterion.from_raw(item) for item in criteria_raw[:5])
                if c is not None
            ] if isinstance(criteria_raw, list) else []
            return StructuredQuery(
                intent=raw.get("intent", "general") if raw.get("intent") in _VALID_INTENTS else "general",
                semantic_query=str(raw.get("semantic_query", "") or "").strip() or _build_semantic_query(brief),
                bm25_keywords=list(raw.get("bm25_keywords", []) or [])[:_MAX_LLM_KEYWORDS],
                hard_filters=HardFilters.from_raw(raw.get("hard_filters")),
                quality_modifier=raw.get("quality_modifier", "any") if raw.get("quality_modifier") in _VALID_QUALITY else "any",
                criteria=criteria,
                raw_brief=brief,
                source=source,
            )
        except Exception as e:
            logger.warning("[QIL] StructuredQuery build failed from %s output: %s", source, e)
            return None

    def _apply_date_floor(sq: StructuredQuery) -> StructuredQuery:
        """
        Regex is primary for dates, the LLM is supplementary (Decisions Log #7,
        amended). An explicit "2022 to 2024" is deterministic and must not depend
        on an inference call that can hallucinate a bound; the LLM's `date_range`
        is kept only where the regex found nothing (e.g. "post-ChatGPT era").

        The LLM's own bounds are additionally discarded when the brief contains no
        temporal wording at all. Observed live: *"diffusion models for text
        generation"* came back with `date_range.from = 2020`, which would have
        silently dropped every pre-2020 paper from a query that never mentioned time.
        """
        year_from, year_to = _extract_date_range(brief)
        if year_from is not None or year_to is not None:
            sq.hard_filters.year_from = year_from
            sq.hard_filters.year_to = year_to
        elif not _TEMPORAL_HINT_RE.search(brief):
            sq.hard_filters.year_from = sq.hard_filters.year_to = None
        return sq

    def _apply_not_terms_floor(sq: StructuredQuery) -> StructuredQuery:
        """
        Union the deterministic exclusion regex into whatever the LLM returned.

        The LLM alone is not reliable here. Observed live: *"... on diffusion
        models. Exclude surveys."* came back with `not_terms = []` **and**
        `"survey"` in `bm25_keywords` — actively searching for the one thing the
        user rejected. The regex catches that phrasing deterministically, so it
        runs on the LLM paths too, not just the rules fallback.
        """
        regex_terms = _extract_not_terms(brief)
        if not regex_terms:
            return sq
        existing = {t.lower() for t in sq.hard_filters.not_terms}
        for term in regex_terms:
            if term not in existing:
                sq.hard_filters.not_terms.append(term)
                existing.add(term)
        # An excluded term must never also be a search term.
        sq.bm25_keywords = [
            kw for kw in sq.bm25_keywords
            if not any(t in kw.lower() or kw.lower() in t for t in existing)
        ]
        return sq

    def _apply_entity_floor(sq: StructuredQuery) -> StructuredQuery:
        """
        Ground authors/venues in the brief, and add any the LLM missed.

        Observed live: the same "by Hinton at NeurIPS" brief returned
        `authors: ["Hinton"]` on one run and `[]` on the next, with "Hinton"
        landing in `bm25_keywords` instead. Both directions are handled — the
        regex supplies what the model missed, and anything the model produced
        that isn't actually in the brief is dropped, since these become hard
        LIKE filters.
        """
        brief_lower = brief.lower()
        hf = sq.hard_filters
        hf.authors = [a for a in hf.authors if _is_grounded_in(a, brief_lower)]
        hf.venues = [v for v in hf.venues if _is_grounded_in(v, brief_lower)]

        known = {a.lower() for a in hf.authors}
        for name in _extract_authors(brief):
            if not any(name.lower() in k or k in name.lower() for k in known):
                hf.authors.append(name)
                known.add(name.lower())
        return sq

    def _apply_quality_guard(sq: StructuredQuery) -> StructuredQuery:
        """
        A quality_modifier must be supported by something the brief actually says.

        Each non-"any" value drives a LanceDB pre-filter (QUALITY_LANCEDB_FILTERS),
        so an inferred one narrows the corpus for a query that never asked. Observed
        live: "diffusion models for text generation" came back `recent`, applying
        `year >= 2023`. The time-based values need temporal wording; "influential"
        needs an impact claim.
        """
        modifier = sq.quality_modifier
        if modifier in {"recent", "emerging", "classic"} and not _TEMPORAL_HINT_RE.search(brief):
            sq.quality_modifier = "any"
        elif modifier == "influential" and not _IMPACT_HINT_RE.search(brief):
            sq.quality_modifier = "any"
        return sq

    def _apply_acronym_floor(sq: StructuredQuery) -> StructuredQuery:
        """
        Correct a wrong acronym expansion in bm25_keywords/semantic_query using
        _ACRONYM_EXPANSIONS. Observed live: "DPO" expanded to "Differentially
        Private Optimization" (wrong — Direct Preference Optimization in alignment
        literature) and "KTO" to "Knowledge Transfer Optimization" (wrong —
        Kahneman-Tversky Optimization); both become literal LanceDB FTS terms.

        Must run before _apply_yake_floor/_apply_not_terms_floor — both mutate
        bm25_keywords and would break the acronym-then-expansion adjacency this
        relies on (empirically observed: the model emits the acronym immediately
        followed by its expansion attempt, when it attempts one at all).

        Only a listed acronym can trigger a correction — an unlisted one is left
        untouched, since Stage A has no way to verify it (no escalation call exists
        yet). Never overwrites a phrase that's already the user's own words in the
        brief — the brief wins where it's explicit, same principle as every other
        floor in this chain.
        """
        brief_lower = brief.lower()
        keywords = sq.bm25_keywords
        for i in range(len(keywords) - 1):
            acronym = keywords[i].strip().upper()
            canonical = _ACRONYM_EXPANSIONS.get(acronym)
            if canonical is None:
                continue
            candidate = keywords[i + 1]
            if candidate.lower() == canonical.lower():
                continue  # already correct
            if not _acronym_letters_match(acronym, candidate):
                continue  # not an expansion attempt at all (e.g. RLHF -> "reward model")
            if candidate.lower() in brief_lower:
                continue  # the user's own words win, even if it reads like an attempt
            keywords[i + 1] = canonical
            if sq.semantic_query and candidate.lower() in sq.semantic_query.lower():
                sq.semantic_query = re.sub(
                    re.escape(candidate), canonical, sq.semantic_query, flags=re.IGNORECASE
                )
        return sq

    def _apply_intent_floor(sq: StructuredQuery) -> StructuredQuery:
        """
        Override the LLM's intent when the rules-path _detect_intent matches a real
        signal and disagrees. Observed live: an 8B model defaults to "novelty" on
        ~70% of test queries regardless of clearer signals — e.g. "give me a broad
        overview..." (rules correctly says survey) came back "novelty" from the LLM.

        _detect_intent's own "general" default carries no signal, so it never
        overrides — only a real signal match (survey/foundational/diversity/novelty
        keyword, or a bare year/named author) is trusted enough to correct the LLM.
        """
        rules_intent = _detect_intent(brief)
        if rules_intent != "general" and rules_intent != sq.intent:
            sq.intent = rules_intent
        return sq

    def _apply_term_quality_floor(sq: StructuredQuery) -> StructuredQuery:
        """
        Strip standalone intent-leak/hypernym terms, and malformed prompt-echo
        artifacts, from bm25_keywords (QIL v3 Stage 1). Observed live: a query
        about diffusion models returned "artificial intelligence", "machine
        learning", "deep learning", "novel", "recent" as keywords — each with
        near-zero discriminative power in a 385K-paper *AI* corpus, verified by
        direct DF measurement (docs/PLAN.md). A live A/B against this corpus's
        own FTS index showed the difference matters: this pruned list scored
        73.3% on-topic in the top hits vs. the unpruned original, on the exact
        query that motivated this floor.

        Also observed live, reproducibly, on the same query: a literal keyword
        "brief: diffusion" — the model echoing the prompt's own Brief-then-quoted-
        text wrapper as if it were extractable content, the same anchoring
        failure mode as R8's leaked worked-example tokens, just a different part
        of the prompt. No legitimate technical term contains a colon, so this is
        safe to catch deterministically rather than needing another prompt rewrite.

        Must run before _apply_yake_floor, which reuses the same acceptance
        check (_is_acceptable_keyword) when topping up — otherwise a stripped
        term could just be re-added.
        """
        sq.bm25_keywords = [kw for kw in sq.bm25_keywords if _is_acceptable_keyword(kw)]
        return sq

    def _apply_verbatim_query_floor(sq: StructuredQuery) -> StructuredQuery:
        """
        Force semantic_query to the brief's own (cleaned) text when the brief is
        short — reusing _build_semantic_query, today only used by the rules
        fallback. Below this length, an LLM paraphrase ("Recent advancements in
        X have led to improved performance...") is invented framing the brief
        never asked for, and it's what SPECTER2 actually encodes for the vector
        arm. Live A/B: verbatim scored 83/100 on-topic vs. 3/100 for the drifted
        paraphrase of the same brief (docs/PLAN.md).

        The 150-char threshold is a judgment call, not literature-derived — the
        literature (Kotte 2026) confirms rewrite harm is real but domain-
        dependent and hard to predict in advance (a purpose-built classifier for
        it only reached AUC 0.593), which is itself the reason not to build
        anything smarter than a length cutoff here.
        """
        if len(brief.strip()) <= _VERBATIM_QUERY_MAX_CHARS:
            sq.semantic_query = _build_semantic_query(brief)
        return sq

    def _apply_floors(sq: StructuredQuery) -> StructuredQuery:
        """Deterministic guards over LLM output — the brief wins where it is explicit."""
        for floor in (
            _apply_acronym_floor, _apply_term_quality_floor, _apply_yake_floor,
            _apply_date_floor, _apply_not_terms_floor, _apply_entity_floor,
            _apply_quality_guard, _apply_intent_floor, _apply_verbatim_query_floor,
        ):
            sq = floor(sq)
        return sq

    def _apply_yake_floor(sq: StructuredQuery) -> StructuredQuery:
        """Supplement bm25_keywords with YAKE if below the minimum floor."""
        if len(sq.bm25_keywords) < _YAKE_FLOOR:
            yake_kws = _extract_keywords(brief, top_n=_YAKE_FLOOR * 2)
            existing_lower = {kw.lower() for kw in sq.bm25_keywords}
            for kw in yake_kws:
                if kw.lower() in existing_lower or not _is_acceptable_keyword(kw):
                    continue
                sq.bm25_keywords.append(kw)
                existing_lower.add(kw.lower())
                if len(sq.bm25_keywords) >= _YAKE_FLOOR:
                    break
        return sq

    # ── Groq path ─────────────────────────────────────────────────────────────
    groq_key = (groq_api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if groq_key:
        raw, err = _call_groq_llm(brief, groq_key, groq_model)
        if raw is not None:
            sq = _build_sq_from_llm(raw, source="llm_groq")
            if sq is not None:
                sq = _apply_floors(sq)
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
                sq = _apply_floors(sq)
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
