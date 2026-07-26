"""
Pipeline orchestrator: runs backend/pipeline_core.py and yields SSE events.

`pipeline_events(req)` is a SYNCHRONOUS generator yielding plain dicts matching
the StageEvent/DoneEvent/ErrorEvent schemas. backend/main.py drives it one step
at a time inside a worker thread, so each heavy stage runs off the event loop
while progress streams to the browser — same pattern as the reference repo's
runner.py, but the actual pipeline calls and stage shape are re-derived from
*this* repo's app.py (Query Intelligence -> LanceDB hybrid RRF -> SPECTER2 ->
CrossEncoder -> highlights -> classify -> Moneyball -> summarize), not copied
from the reference's stale SQLite/BM25 version (see docs/PLAN.md's F-01..F-06
"Critical mismatch" note).

select_embedding_candidates (Stage 0-4) runs as ONE call with an internal
`emit(msg, level)` callback rather than separate top-level calls, so getting
live per-stage SSE progress out of it needs a producer-thread + queue bridge
(`_stream_call`) instead of just yielding between calls like the other steps.

Stage mapping (to the frontend's 7-step loader — see frontend/src/App.jsx STEPS):
  0 Fetch & Filter     -> LanceDB date fetch + NOT-term filter
  1 Hybrid Retrieval   -> QIL (Stage 0) + LanceDB FTS+vector RRF (Stage 1)
  2 Semantic Rerank    -> SPECTER2 / MiniLM fallback (Stage 2)
  3 Precision Rerank   -> CrossEncoder + abstract highlights (Stage 3-4)
  4 Classify           -> scibert_classify_papers (CrossEncoder threshold / heuristic fallback)
  5 Impact Score        -> Moneyball citation scoring (LLM) or heuristic
  6 Summarize          -> plain-English summaries for the top N (LLM providers only)
"""
import hashlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, List, Optional, Tuple

from . import pipeline_core as pc
from .models import PaperOut, QueryUnderstanding

# Persistent QIL cache across requests within this process — same role as
# app.py's st.session_state["_qil_cache"], just process-scoped instead of
# per-Streamlit-session (this backend has no per-user session concept yet).
_QIL_CACHE: Dict[str, Any] = {}

_STAGE_MARKERS: Tuple[Tuple[str, int], ...] = (
    ("Stage 4", 3),
    ("Cross-Encoder", 3),
    ("Stage 3", 3),
    ("MiniLM Stage 2", 2),
    ("SPECTER2", 2),
    ("Stage 2", 2),
    ("RRF Stage 1", 1),
    ("LanceDB unavailable", 1),
    ("Stage 1", 1),
    ("Stage 0", 1),
)


def _classify_stage_message(msg: str) -> int:
    for marker, idx in _STAGE_MARKERS:
        if marker in msg:
            return idx
    return 1  # default bucket: the generic "Starting hybrid search..." opener


def _stream_call(fn, *args, **kwargs) -> Generator[Tuple[str, Any, Optional[str]], None, None]:
    """
    Run fn(*args, emit=<queue-pushing callback>, **kwargs) in a background
    thread, yielding ("msg", text, level) for each emit() call as it happens,
    then finally ("result", return_value, None). Re-raises fn's exception (if
    any) in the caller's thread after the worker finishes.
    """
    q: "queue.Queue" = queue.Queue()
    outcome: Dict[str, Any] = {}

    def _emit(msg: str, level: str = "write") -> None:
        q.put(("msg", msg, level))

    def _worker() -> None:
        try:
            outcome["value"] = fn(*args, emit=_emit, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised in the caller below
            outcome["error"] = exc
        finally:
            q.put(("__done__", None, None))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        kind, msg, level = q.get()
        if kind == "__done__":
            break
        yield (kind, msg, level)

    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    yield ("result", outcome.get("value"), None)


def _predict_citations_with_emit(used_papers, llm_config, quality_modifier, emit):
    """Adapts predict_citations_direct's (done, total) on_progress into _stream_call's
    emit(msg, level) — T-68: without this, Stage 5's concurrent I/O phase ran silently,
    the UI showing no change for the entire wait. Throttled to every 10th paper + the
    last one, matching how the two calls per paper compound into progress ticks."""
    def on_progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            emit(f"Impact Score: {done}/{total} papers scored", "info")

    return pc.predict_citations_direct(
        used_papers, llm_config, quality_modifier=quality_modifier, on_progress=on_progress,
    )


def _provider_to_internal(provider: str) -> str:
    return "free_local" if provider == "free" else provider


# Default base URL per provider. openrouter/ollama_local/ollama_cloud all speak
# the OpenAI-compatible chat-completions API, so call_llm() routes them through
# the same "openai" client branch as openai itself -- only the base_url differs.
_PROVIDER_API_BASE = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama_local": "http://localhost:11434/v1",
    "ollama_cloud": "https://ollama.com/v1",
}


def _make_llm_config(req) -> pc.LLMConfig:
    provider = _provider_to_internal(req.provider)
    api_base = _PROVIDER_API_BASE.get(provider, "")
    # ollama_local needs no real key -- the OpenAI-compatible endpoint doesn't
    # validate it, but call_llm()'s "no key -> skip LLM call" guard would
    # otherwise treat this provider as unconfigured.
    api_key = req.api_key or ("ollama" if provider == "ollama_local" else "")
    return pc.LLMConfig(
        api_key=api_key,
        model=req.model or "",
        api_base=api_base,
        provider=provider,
    )


def _clean_bullets(bullets) -> List[str]:
    # Moneyball explanations contain **markdown** bold; strip it for the plain
    # React rendering (emojis kept, same as the reference repo's runner).
    return [b.replace("**", "").strip() for b in (bullets or [])]


def _to_paper_out(p: "pc.Paper", rank: int, summary: Optional[str]) -> PaperOut:
    pred = p.predicted_citations
    too_new = pred == -1.0
    relevance = p.llm_relevance_score if p.llm_relevance_score is not None else (p.semantic_relevance or 0.0)
    relevance = max(0.0, min(1.0, float(relevance)))
    return PaperOut(
        rank=rank,
        arxiv_id=p.arxiv_id,
        title=p.title,
        authors=p.authors or [],
        venue=p.venue,
        abstract=p.abstract,
        arxiv_url=p.arxiv_url,
        pdf_url=p.pdf_url,
        score=(-1.0 if too_new else float(pred or 0.0)),
        too_new=too_new,
        focus=(p.focus_label if p.focus_label in ("primary", "secondary", "off-topic") else "off-topic"),
        relevance=relevance,
        relevance_basis=(p.relevance_basis if p.relevance_basis == "cross_encoder" else "embedding"),
        evidence=p.matched_sentences or [],
        why=_clean_bullets(p.prediction_explanations),
        summary=summary,
    )


def _to_query_understanding(sq) -> Optional[QueryUnderstanding]:
    """Builds the "How your query was read" payload straight from what QIL
    produced — deliberately just the fields that exist (intent, search terms,
    excluded terms, quality modifier), not a per-paper rubric. See
    docs/asta-ui-comparison-design.md §3 for why the two are not the same thing."""
    if sq is None:
        return None
    return QueryUnderstanding(
        intent=sq.intent,
        search_terms=list(sq.bm25_keywords),
        excluded_terms=list(sq.hard_filters.get("not_terms", [])),
        quality_modifier=sq.quality_modifier,
        source=sq.source,
    )


def _sort_group(group: List["pc.Paper"]) -> List["pc.Paper"]:
    """Scored (high->low citation impact) then unscored/"too new" (high->low relevance) — mirrors app.py's sort_group."""
    scored = [p for p in group if p.predicted_citations is not None and p.predicted_citations >= 0]
    unscored = [p for p in group if p.predicted_citations == -1.0]
    scored.sort(key=lambda p: p.predicted_citations, reverse=True)
    unscored.sort(key=lambda p: (p.llm_relevance_score or 0.0, p.semantic_relevance or 0.0), reverse=True)
    return scored + unscored


def _stage(index: int, status: str, name: str, detail: str = "") -> Dict[str, Any]:
    return {"type": "stage", "index": index, "status": status, "name": name, "detail": detail}


STAGE_NAMES = [
    "Fetch & Filter",
    "Hybrid Retrieval",
    "Semantic Rerank",
    "Precision Rerank",
    "Classify",
    "Impact Score",
    "Summarize",
]


def pipeline_events(req):
    """Synchronous generator yielding SSE event dicts. See module docstring for the stage mapping."""
    t0 = time.perf_counter()
    is_llm = req.provider in ("openai", "gemini", "groq", "anthropic", "openrouter", "ollama_local", "ollama_cloud")
    llm_config = _make_llm_config(req)

    brief = (req.query or "").strip()
    not_text = (req.exclude or "").strip()
    query_brief = pc.build_query_brief(brief, not_text)
    start_date, end_date = pc.get_date_range(req.date_range)

    # ---- Stage 0: fetch + NOT-term filter ----
    yield _stage(0, "start", STAGE_NAMES[0])
    current = pc.fetch_papers_from_lancedb(start_date, end_date)
    if not_text:
        current, removed = pc.filter_papers_by_not_terms(current, not_text)
    else:
        removed = 0
    if not current:
        yield {"type": "error", "message": "No papers found in the corpus for that date range."}
        return
    yield _stage(0, "done", STAGE_NAMES[0], f"{len(current):,} papers" + (f" ({removed} excluded)" if removed else ""))

    # ---- Stage 1-3 (UI): select_embedding_candidates' internal Stage 0-4 ----
    qil_key = hashlib.md5(query_brief.strip().lower().encode()).hexdigest()
    current_ui_stage = 1
    yield _stage(current_ui_stage, "start", STAGE_NAMES[current_ui_stage])
    candidates: List[pc.Paper] = []
    for kind, payload, _level in _stream_call(
        pc.select_embedding_candidates,
        current,
        query_brief,
        llm_config=llm_config,
        max_candidates=150,
        qil_cache=_QIL_CACHE,
    ):
        if kind == "msg":
            target_stage = _classify_stage_message(payload)
            if target_stage != current_ui_stage:
                yield _stage(current_ui_stage, "done", STAGE_NAMES[current_ui_stage])
                current_ui_stage = target_stage
                yield _stage(current_ui_stage, "start", STAGE_NAMES[current_ui_stage])
            else:
                yield _stage(current_ui_stage, "start", STAGE_NAMES[current_ui_stage], payload)
        elif kind == "result":
            candidates = payload or []
    yield _stage(current_ui_stage, "done", STAGE_NAMES[current_ui_stage], f"{len(candidates)} candidates")

    # select_embedding_candidates writes its StructuredQuery into _QIL_CACHE under
    # this same key (pipeline_core.py's Stage 0) — fetched once here and reused
    # below, regardless of provider (QIL's own Groq/OpenRouter key is independent
    # of the reasoning provider the user picked).
    cached_sq = _QIL_CACHE.get(qil_key)
    query_understanding = _to_query_understanding(cached_sq)

    if not candidates:
        candidates = current  # same fallback app.py uses when embedding stage returns nothing

    # ---- Stage 4: relevance classification (CrossEncoder threshold, provider-agnostic) ----
    yield _stage(4, "start", STAGE_NAMES[4])
    candidates = pc.scibert_classify_papers(candidates)
    primary_papers = [p for p in candidates if p.focus_label == "primary"]
    secondary_papers = [p for p in candidates if p.focus_label == "secondary"]
    for group in (primary_papers, secondary_papers):
        group.sort(key=lambda p: (p.llm_relevance_score or 0.0, p.semantic_relevance or 0.0), reverse=True)
    yield _stage(4, "done", STAGE_NAMES[4], f"{len(primary_papers)} primary, {len(secondary_papers)} secondary")

    # ---- Build the citation-impact scoring set (mirrors app.py step 5) ----
    if primary_papers:
        used_papers = primary_papers.copy()
        if len(primary_papers) < pc.MIN_FOR_PREDICTION and secondary_papers:
            used_papers.extend(secondary_papers[: pc.MIN_FOR_PREDICTION - len(primary_papers)])
    elif secondary_papers:
        used_papers = secondary_papers[:20]
    else:
        yield {"type": "error", "message": "No relevant papers found for that brief."}
        return

    # ---- Stage 5: Moneyball impact scoring ----
    yield _stage(5, "start", STAGE_NAMES[5])
    if is_llm:
        quality_modifier = cached_sq.quality_modifier if cached_sq else "any"
        for kind, payload, _level in _stream_call(
            _predict_citations_with_emit, used_papers, llm_config, quality_modifier,
        ):
            if kind == "msg":
                yield _stage(5, "start", STAGE_NAMES[5], payload)
            elif kind == "result":
                used_papers = payload
    else:
        used_papers = pc.assign_heuristic_citations_free(used_papers)

    primaries = [p for p in used_papers if p.focus_label == "primary"]
    secondaries = [p for p in used_papers if p.focus_label == "secondary"]
    others = [p for p in used_papers if p.focus_label not in ("primary", "secondary")]
    ranked = _sort_group(primaries) + _sort_group(secondaries) + _sort_group(others)
    yield _stage(5, "done", STAGE_NAMES[5], f"{len(ranked)} papers scored")

    # ---- Stage 6: plain-English summaries for the top N (LLM providers only) ----
    yield _stage(6, "start", STAGE_NAMES[6])
    top_n = min(req.top_n, len(ranked))
    summaries: Dict[str, Optional[str]] = {}
    if is_llm:
        def _summarize(p: "pc.Paper") -> Tuple[str, Optional[str]]:
            try:
                return p.arxiv_id, pc.summarize_paper_plain_english(p, llm_config)
            except Exception:
                return p.arxiv_id, None

        # T-69: same serial-per-paper LLM pattern as Stage 5, smaller scale (top_n <= 10)
        # — reuses Stage 5's LLM concurrency budget rather than a separate pool.
        with ThreadPoolExecutor(max_workers=pc.LLM_MAX_CONCURRENCY) as pool:
            for arxiv_id, summary in pool.map(_summarize, ranked[:top_n]):
                summaries[arxiv_id] = summary
    yield _stage(6, "done", STAGE_NAMES[6], f"Top {top_n} summarized" if is_llm else "Heuristic mode — no summaries")

    papers_out = [_to_paper_out(p, i + 1, summaries.get(p.arxiv_id)) for i, p in enumerate(ranked)]
    yield {
        "type": "done",
        "papers": [po.model_dump() for po in papers_out],
        "primary_count": len(primaries),
        "secondary_count": len(secondaries),
        "total_seconds": round(time.perf_counter() - t0, 1),
        "provider": req.provider,
        "query_understanding": query_understanding.model_dump() if query_understanding else None,
    }
