"""
tests/test_pipeline_core_port.py

F-02: proves backend/pipeline_core.py is a working Streamlit-free port of
app.py's pipeline — importable with zero `streamlit` reference, and its pure
logic (no network/models) matches app.py's behavior. Fast/offline, mirrors
tests/test_select_embedding_candidates.py's no-network style.
"""
from backend import pipeline_core as pc


def test_no_streamlit_imported():
    """pipeline_core must not bind `streamlit`/`st` at module level (no `import streamlit as st`).

    Checked via pc.__dict__ rather than sys.modules: other test modules in the
    same pytest run import app.py (which does import streamlit), so sys.modules
    would contain "streamlit" regardless of what pipeline_core itself does.
    """
    assert "st" not in pc.__dict__
    assert "streamlit" not in pc.__dict__


def test_get_secret_env_only(monkeypatch):
    monkeypatch.setenv("PIPELINE_CORE_TEST_KEY", "  'value'  ")
    assert pc._get_secret("PIPELINE_CORE_TEST_KEY") == "value"
    assert pc._get_secret("PIPELINE_CORE_TEST_KEY_UNSET", "fallback") == "fallback"


def test_parse_and_filter_not_terms():
    papers = [
        pc.Paper(
            arxiv_id="1", title="Diffusion models for video generation",
            authors=[], email_domains=[], abstract="video synthesis",
            submitted_date=None, pdf_url="", arxiv_url="",
        ),
        pc.Paper(
            arxiv_id="2", title="Diffusion models for images",
            authors=[], email_domains=[], abstract="image synthesis",
            submitted_date=None, pdf_url="", arxiv_url="",
        ),
    ]
    filtered, removed = pc.filter_papers_by_not_terms(papers, "video generation")
    assert removed == 1
    assert [p.arxiv_id for p in filtered] == ["2"]


def test_resolve_moneyball_weights_quality_override():
    assert pc.resolve_moneyball_weights("influential") == pc.QUALITY_MONEYBALL_WEIGHTS["influential"]
    assert pc.resolve_moneyball_weights("any") == pc.DEFAULT_MONEYBALL_WEIGHTS


def test_cache_resource_replaced_by_lru_cache():
    """Model/connection loaders must expose cache_clear() (functools.lru_cache), not @st.cache_resource."""
    for loader in (pc.get_local_embed_model, pc.get_lancedb_table, pc.get_specter2_model, pc.get_cross_encoder_model):
        assert hasattr(loader, "cache_clear")


def test_select_embedding_candidates_signature_matches_app():
    """Same emit/qil_cache seam as app.py's decoupled version (2026-07-16 architecture deepening)."""
    import inspect
    params = list(inspect.signature(pc.select_embedding_candidates).parameters)
    assert "emit" in params
    assert "qil_cache" in params


def test_join_where_composes_and_short_circuits():
    assert pc._join_where(None, None) is None
    assert pc._join_where("year >= 2020") == "(year >= 2020)"
    assert pc._join_where("a", None, "b") == "(a) AND (b)"


# ─── R6/R7: brief-sourced filters must be visible, not just applied ───────────

def test_query_understanding_surfaces_brief_filters():
    """A filter narrowing the search that the user can't see is one they can't correct."""
    from backend.runner import _to_query_understanding, _date_window_label
    from query_intelligence import HardFilters, StructuredQuery

    sq = StructuredQuery(
        intent="specific",
        bm25_keywords=["diffusion"],
        quality_modifier="influential",
        source="rules",
        hard_filters=HardFilters(
            not_terms=["surveys"], authors=["Hinton"], venues=["NeurIPS"],
            year_from=2022, year_to=2024,
        ),
    )
    qu = _to_query_understanding(sq)
    assert qu.date_window == "2022–2024"
    assert qu.authors == ["Hinton"]
    assert qu.venues == ["NeurIPS"]
    assert qu.excluded_terms == ["surveys"]

    assert _date_window_label(HardFilters(year_from=2020)) == "from 2020"
    assert _date_window_label(HardFilters(year_to=2020)) == "up to 2020"
    assert _date_window_label(HardFilters()) is None


def test_query_understanding_surfaces_criteria():
    """QIL v3 Stage 2: criteria are display-only, projected straight through."""
    from backend.runner import _to_query_understanding
    from query_intelligence import Criterion, StructuredQuery

    sq = StructuredQuery(
        intent="specific",
        bm25_keywords=["sparse attention"],
        source="llm_groq",
        criteria=[
            Criterion(name="Mechanism", definition="Proposes a new sparse attention mechanism.", strength="must"),
            Criterion(name="Benchmark", definition="Includes long-context benchmarks.", strength="should"),
        ],
    )
    qu = _to_query_understanding(sq)
    assert len(qu.criteria) == 2
    assert qu.criteria[0].name == "Mechanism"
    assert qu.criteria[0].strength == "must"
    assert qu.criteria[1].strength == "should"


def test_query_understanding_criteria_empty_by_default():
    from backend.runner import _to_query_understanding
    from query_intelligence import StructuredQuery

    qu = _to_query_understanding(StructuredQuery(source="rules"))
    assert qu.criteria == []
