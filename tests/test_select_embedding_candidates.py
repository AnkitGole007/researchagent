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
from query_intelligence import HardFilters


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
        hard_filters = HardFilters()
        rrf_weight_bm25 = 1.0
        rrf_weight_faiss = 1.0

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


# ─── R4 / Gap 9: intent-derived RRF weights reach Stage 1 ─────────────────────

class _FakeQuery:
    """Every builder method returns self; to_list() yields the canned rows.

    Records the WHERE clause on the owning table and returns nothing when that
    clause contains the table's `block_if` marker — how the relax ladder is tested.
    """

    def __init__(self, table, rows):
        self._table, self._rows = table, rows
        self._where = None

    def select(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def where(self, clause=None, **k):
        self._where = clause
        return self

    def to_list(self):
        self._table.wheres.append(self._where)
        if self._table.block_if and self._where and self._table.block_if in self._where:
            return []
        return self._rows


class _FakeTable:
    def __init__(self, fts_rows, vec_rows, block_if=None):
        self._fts_rows, self._vec_rows = fts_rows, vec_rows
        self.block_if = block_if
        self.wheres = []

    def search(self, query, query_type=None, **k):
        return _FakeQuery(self, self._fts_rows if query_type == "fts" else self._vec_rows)


def _rank_ids_with_weights(w_fts: float, w_vec: float):
    """Rank an FTS-only paper (A) against a vector-only paper (B) under given weights."""
    import numpy as np

    papers = [make_paper("A"), make_paper("B")]
    table = _FakeTable(
        fts_rows=[{"arxiv_id": "A", "has_embedding": False, "vector": None}],
        vec_rows=[{"arxiv_id": "B", "vector": None}],
    )
    ranked, _ = pc._lancedb_hybrid_stage1(
        table, papers, "query", np.zeros(768), top_k=10, w_fts=w_fts, w_vec=w_vec
    )
    return [p.arxiv_id for p in ranked]


def test_rrf_weights_shift_ranking_toward_the_favoured_arm():
    # foundational-style intent: FTS weighted up, vector down -> FTS-only paper wins
    assert _rank_ids_with_weights(1.4, 0.8)[0] == "A"
    # novelty-style intent: the mirror image -> vector-only paper wins
    assert _rank_ids_with_weights(0.8, 1.4)[0] == "B"


def test_rrf_default_weights_are_symmetric():
    """1.0/1.0 must reproduce plain RRF — A and B are mirror cases, so scores tie."""
    import numpy as np

    papers = [make_paper("A"), make_paper("B")]
    table = _FakeTable(
        fts_rows=[{"arxiv_id": "A", "has_embedding": False, "vector": None}],
        vec_rows=[{"arxiv_id": "B", "vector": None}],
    )
    ranked, _ = pc._lancedb_hybrid_stage1(table, papers, "query", np.zeros(768), top_k=10)
    assert {p.arxiv_id for p in ranked} == {"A", "B"}
    assert ranked[0].rrf_score == ranked[1].rrf_score


# ─── R6/R7: brief-sourced filters and the relax ladder ────────────────────────

def test_brief_filters_reach_stage1_then_relax_when_the_pool_starves():
    """Entity filter is tried first, then dropped so the search still returns."""
    import numpy as np
    from query_intelligence import HardFilters

    papers = [make_paper(str(i)) for i in range(60)]
    table = _FakeTable(
        fts_rows=[{"arxiv_id": str(i), "has_embedding": False, "vector": None} for i in range(60)],
        vec_rows=[],
        block_if="authors LIKE",  # the author filter matches nothing in this corpus
    )

    class _SQ:
        source = "rules"
        intent = "specific"
        quality_modifier = "any"
        bm25_keywords = ["test"]
        bm25_query_string = "test"
        semantic_query = "test query"
        rrf_weight_bm25 = 1.0
        rrf_weight_faiss = 1.0
        hard_filters = HardFilters(authors=["Hinton"], year_from=2020)

    with patch.object(pc, "analyse_query", lambda **kw: _SQ()), \
         patch.object(pc, "get_lancedb_table", return_value=table), \
         patch.object(pc, "cross_encoder_rerank", side_effect=lambda ps, q, n3: ps), \
         patch.object(pc, "extract_abstract_highlights", side_effect=lambda ps, q: ps), \
         patch.object(pc, "enrich_paper_signals", side_effect=lambda ps: ps), \
         patch.object(pc, "specter2_vector_rerank", return_value=[]), \
         patch.object(pc, "minilm_vector_rerank", side_effect=lambda ps, q, n2: ps):
        result = pc.select_embedding_candidates(papers, "brief", emit=lambda *a: None)

    tried = [w for w in table.wheres if w]
    assert any("authors LIKE '%Hinton%'" in w for w in tried), "author filter never applied"
    assert any("year >= 2020" in w for w in tried), "brief date window never applied"
    # ...and the ladder relaxed the entity filter while keeping the year bound.
    assert any("authors LIKE" not in w and "year >= 2020" in w for w in tried), \
        "entity filter was never relaxed"
    assert len(result) > 0, "relax ladder should have recovered a non-empty result"
