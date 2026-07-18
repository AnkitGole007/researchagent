"""
tests/test_select_embedding_candidates.py

Proves the deepening: select_embedding_candidates takes an injectable `emit`
callback and a plain `qil_cache` dict instead of calling st.write/st.info/
st.warning and st.session_state directly. Stub every collaborator (LanceDB,
SPECTER2, CrossEncoder, QIL) so this runs fast, with no Streamlit runtime,
no network, no models — exactly what the coupling used to block.

Imports from backend.pipeline_core (F-02's port), not app.py — app.py was
retired in F-06 once the FastAPI/React stack replaced the Streamlit UI.
"""
from datetime import datetime
from unittest.mock import patch

from backend import pipeline_core as pc


def make_paper(arxiv_id: str) -> pc.Paper:
    return pc.Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors=["Author"],
        email_domains=[],
        abstract="Abstract text.",
        submitted_date=datetime(2025, 1, 1),
        pdf_url="",
        arxiv_url="",
    )


def test_runs_without_streamlit_session_state():
    """No st.session_state access when qil_cache/emit are injected explicitly."""
    papers = [make_paper("1"), make_paper("2")]
    messages = []

    def collect(msg, level="write"):
        messages.append((level, msg))

    with patch.object(pc, "analyse_query", None), \
         patch.object(pc, "get_lancedb_table", return_value=None), \
         patch.object(pc, "cross_encoder_rerank", side_effect=lambda ps, q, n3: ps), \
         patch.object(pc, "extract_abstract_highlights", side_effect=lambda ps, q: ps), \
         patch.object(pc, "enrich_paper_signals", side_effect=lambda ps: ps), \
         patch.object(pc, "specter2_vector_rerank", return_value=[]), \
         patch.object(pc, "minilm_vector_rerank", side_effect=lambda ps, q, n2: ps):
        result = pc.select_embedding_candidates(
            papers, "test query", emit=collect, qil_cache={},
        )

    assert result == papers
    assert any("LanceDB unavailable" in msg for _, msg in messages)
    assert any(level == "warning" for level, _ in messages)


def test_qil_cache_persists_across_calls():
    """Passing the same dict across two calls reuses the cached QIL result."""
    papers = [make_paper("1")]
    cache: dict = {}
    call_count = {"n": 0}

    class _FakeSQ:
        source = "rules"
        intent = "novelty"
        quality_modifier = "recent"
        bm25_keywords = ["test"]
        bm25_query_string = "test"
        semantic_query = "test query"
        hard_filters = {}

    def fake_analyse_query(**kwargs):
        call_count["n"] += 1
        return _FakeSQ()

    with patch.object(pc, "analyse_query", fake_analyse_query), \
         patch.object(pc, "get_lancedb_table", return_value=None), \
         patch.object(pc, "cross_encoder_rerank", side_effect=lambda ps, q, n3: ps), \
         patch.object(pc, "extract_abstract_highlights", side_effect=lambda ps, q: ps), \
         patch.object(pc, "enrich_paper_signals", side_effect=lambda ps: ps), \
         patch.object(pc, "specter2_vector_rerank", return_value=[]), \
         patch.object(pc, "minilm_vector_rerank", side_effect=lambda ps, q, n2: ps):
        pc.select_embedding_candidates(papers, "same query", emit=lambda *a: None, qil_cache=cache)
        pc.select_embedding_candidates(papers, "same query", emit=lambda *a: None, qil_cache=cache)

    assert call_count["n"] == 1, "second call should hit qil_cache, not re-run analyse_query"
