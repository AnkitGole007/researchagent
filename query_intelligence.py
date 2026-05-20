"""
query_intelligence.py — Task P-00: Query Intelligence Layer

Sits before Stage 1 of the 3-stage hybrid retrieval pipeline.
Converts a raw free-text research brief into three structured outputs:

  1. semantic_query  → clean 1-2 sentence statement for FAISS / SPECTER2
  2. bm25_keywords   → 5-10 precise terms for BM25
  3. intent          → drives RRF weights and Stage 1 adaptive behaviour
  4. hard_filters    → metadata constraints (date_range, not_terms)
  5. quality_modifier → recency / influence signal

Primary path:  Groq LLM (llama-3.3-70b-versatile) with JSON output.
Fallback path: Rules-based extractor — no API key required.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Intent = Literal["novelty", "diversity", "foundational", "specific", "survey", "general"]

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
        (FAISS / SPECTER2). HyDE is retired when this is populated.
    bm25_keywords : List[str]
        5–10 precise terms including synonyms and domain aliases.
        Replaces the raw ``query_brief`` sent to BM25.
    hard_filters : dict
        Metadata constraints extracted from the brief.
        Keys: ``date_range`` (Optional[tuple[str,str]]), ``not_terms`` (List[str]),
              ``authors`` (List[str]), ``venues`` (List[str]).
    quality_modifier : str
        One of "recent" | "influential" | "emerging" | "classic" | "any".
    raw_brief : str
        Original unmodified brief for logging / debug.
    source : str
        "llm" | "rules" — indicates which path produced this object.
    """
    intent: Intent = "general"
    semantic_query: str = ""
    bm25_keywords: List[str] = field(default_factory=list)
    hard_filters: Dict[str, Any] = field(default_factory=dict)
    quality_modifier: str = "any"
    raw_brief: str = ""
    source: str = "rules"

    # ── Convenience: intent → RRF weight override ──────────────────────────
    @property
    def rrf_weight_faiss(self) -> float:
        """Higher FAISS weight for semantic / novelty / survey intents."""
        return {
            "novelty":     1.4,
            "survey":      1.2,
            "diversity":   1.2,
            "foundational": 0.8,
            "specific":    1.0,
            "general":     1.0,
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
# LLM-backed analyser
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a research query analyser. Given a user's research brief, "
    "decompose it into structured components. "
    "Respond ONLY with a single valid JSON object — no markdown, no explanation."
)

_LLM_PROMPT_TEMPLATE = """Research brief:
\"\"\"
{brief}
\"\"\"

Produce a JSON object with these exact keys:
{{
  "intent": "<one of: novelty | diversity | foundational | specific | survey | general>",
  "semantic_query": "<clean 1-2 sentence description optimised for dense vector search>",
  "bm25_keywords": ["<5-10 precise terms, synonyms, domain aliases>"],
  "hard_filters": {{
    "not_terms": ["<explicit exclusion terms mentioned by user>"],
    "authors": [],
    "venues": []
  }},
  "quality_modifier": "<one of: recent | influential | emerging | classic | any>"
}}

Rules:
- intent=novelty if the user explicitly wants new/novel/recent/emerging work.
- intent=foundational if the user wants influential/classic/seminal/foundational work.
- intent=survey if the user wants overviews, comparisons, or comprehensive studies.
- intent=specific if the user names a specific paper, author, or narrow technique.
- intent=diversity if the user wants broad coverage across sub-areas.
- Otherwise intent=general.
- bm25_keywords must include synonyms that BM25 would match (e.g. "LLM" and "large language model").
- semantic_query must be a descriptive statement about the paper's contribution, NOT a question.
- hard_filters.not_terms should only include terms explicitly rejected in the brief.
"""


def _call_groq_llm(brief: str, api_key: str, model: str = "llama-3.3-70b-versatile") -> Optional[dict]:
    """
    Single Groq API call with structured JSON output.
    Returns the parsed dict or None on any failure.
    """
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        logger.debug("[QIL] groq package not installed — skip LLM path")
        return None

    prompt = _LLM_PROMPT_TEMPLATE.format(brief=brief[:3000])  # truncate to avoid token blowout

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
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip optional code fences
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(
                    l for l in lines
                    if not l.strip().startswith("```")
                ).strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("[QIL] JSON parse error (attempt %d): %s", attempt + 1, e)
        except Exception as e:
            logger.warning("[QIL] Groq call error (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1)

    return None


# ---------------------------------------------------------------------------
# Rules-based fallback (no API key)
# ---------------------------------------------------------------------------

# Common NLP stop-words to strip before extracting BM25 keywords
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

# Intent keyword heuristics
_INTENT_SIGNALS: Dict[str, List[str]] = {
    "novelty":      ["novel", "new", "recent", "latest", "emerging", "state-of-the-art", "sota", "cutting-edge"],
    "foundational": ["foundational", "seminal", "classic", "influential", "survey textbook", "original", "pioneering"],
    "survey":       ["survey", "overview", "comprehensive", "review", "comparison", "benchmark", "evaluation study"],
    "specific":     [],  # detected below by named entities (author names, years, paper titles)
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
    # Specific: year or "by <Name>" pattern
    if re.search(r"\b(20\d{2})\b", brief_lower) or re.search(r"\bby [A-Z][a-z]+", brief_lower):
        return "specific"
    return "general"


def _extract_keywords(brief: str, top_n: int = 8) -> List[str]:
    """
    Lightweight TF-IDF inspired extraction: rank all candidate n-grams by length
    (longer = more specific) and dedup. Returns top_n terms.
    No external libraries required.
    """
    text = brief.lower()
    # Bigrams and unigrams
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z\-]+\b", text)
    filtered = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]

    # Frequency count
    freq: Dict[str, int] = {}
    for t in filtered:
        freq[t] = freq.get(t, 0) + 1

    # Bigrams from original brief
    words = text.split()
    bigrams: List[str] = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 not in _STOPWORDS and w2 not in _STOPWORDS and len(w1) > 2 and len(w2) > 2:
            bigram = f"{w1} {w2}"
            bigrams.append(bigram)

    # Prefer bigrams (more specific) then unigrams by frequency
    bigram_freq: Dict[str, int] = {}
    for bg in bigrams:
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1

    # Combine: give bigrams a score boost of 2
    combined: Dict[str, float] = {}
    for bg, c in bigram_freq.items():
        combined[bg] = c * 2.0
    for ug, c in freq.items():
        # Don't add unigram if it's already part of a bigram
        if not any(ug in bg for bg in bigram_freq):
            combined[ug] = float(c)

    sorted_terms = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_terms[:top_n]]


def _build_semantic_query(brief: str) -> str:
    """
    Strip NOT-looking-for sections, trim stop-phrases, produce a clean 1-2 sentence statement.
    """
    # Remove "WHAT I AM NOT LOOKING FOR" sections (from build_query_brief format)
    brief_clean = re.split(r"WHAT I AM NOT LOOKING FOR", brief, flags=re.IGNORECASE)[0]
    # Remove "RESEARCH BRIEF:" prefix
    brief_clean = re.sub(r"^RESEARCH BRIEF:\s*", "", brief_clean, flags=re.MULTILINE).strip()
    # Collapse whitespace
    brief_clean = re.sub(r"\s+", " ", brief_clean).strip()
    # Hard cap at 512 chars for embedding model input
    if len(brief_clean) > 512:
        # Try to cut at sentence boundary
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
    # Split on commas, newlines, semicolons
    raw_terms = re.split(r"[,\n;]+", not_section)
    return [t.strip().lower() for t in raw_terms if len(t.strip()) > 3][:10]


def _rules_based_analyse(brief: str) -> StructuredQuery:
    """Pure-Python fallback analyser. No external APIs."""
    brief_lower = brief.lower()
    intent = _detect_intent(brief_lower)

    # Quality modifier
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
    groq_model: str = "llama-3.3-70b-versatile",
) -> StructuredQuery:
    """
    Analyse a raw research brief and return a :class:`StructuredQuery`.

    Primary path: Groq LLM call (fast, ~200-400 ms, fraction of a cent).
    Fallback path: Rules-based extractor (zero cost, no network).

    Parameters
    ----------
    brief : str
        The raw free-text research brief from the user.
    groq_api_key : str, optional
        If provided, the Groq API will be used; otherwise rules-based fallback runs.
    groq_model : str
        Groq model name (default: llama-3.3-70b-versatile).

    Returns
    -------
    StructuredQuery
    """
    if not brief or not brief.strip():
        logger.debug("[QIL] Empty brief — returning empty StructuredQuery")
        return StructuredQuery(raw_brief=brief, source="rules")

    t0 = time.perf_counter()

    # ── LLM path ──────────────────────────────────────────────────────────
    key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    if key and key.strip():
        raw = _call_groq_llm(brief, key, groq_model)
        if raw and isinstance(raw, dict):
            try:
                sq = StructuredQuery(
                    intent=raw.get("intent", "general") if raw.get("intent") in _VALID_INTENTS else "general",
                    semantic_query=str(raw.get("semantic_query", "") or "").strip() or _build_semantic_query(brief),
                    bm25_keywords=list(raw.get("bm25_keywords", []) or [])[:10],
                    hard_filters=raw.get("hard_filters", {"not_terms": [], "authors": [], "venues": []}),
                    quality_modifier=raw.get("quality_modifier", "any") if raw.get("quality_modifier") in _VALID_QUALITY else "any",
                    raw_brief=brief,
                    source="llm",
                )
                # Ensure bm25_keywords is never empty
                if not sq.bm25_keywords:
                    sq.bm25_keywords = _extract_keywords(brief)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("[QIL] LLM path: intent=%s quality=%s keywords=%s (%.0f ms)",
                            sq.intent, sq.quality_modifier, sq.bm25_keywords[:3], elapsed)
                return sq
            except Exception as e:
                logger.warning("[QIL] Failed to build StructuredQuery from LLM output: %s", e)

    # ── Rules-based fallback ───────────────────────────────────────────────
    sq = _rules_based_analyse(brief)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[QIL] Rules path: intent=%s quality=%s keywords=%s (%.0f ms)",
                sq.intent, sq.quality_modifier, sq.bm25_keywords[:3], elapsed)
    return sq
