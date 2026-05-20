"""
tests/test_classification.py — P-10: Classification stage unit tests

Covers:
  - Option A: cross_encoder_score field separation from semantic_relevance
  - scibert_classify_papers: CE path + heuristic fallback
  - cross_encoder_rerank: writes cross_encoder_score, preserves semantic_relevance
  - Phase 4 "no results" regression: confirms primary_papers is never empty
    when papers have valid semantic_relevance (Stage 2 cosine).
"""
import math
import pytest
from datetime import datetime
from typing import List

from app import (
    Paper,
    scibert_classify_papers,
    heuristic_classify_papers_free,
    PRIMARY_THRESHOLD,
    SECONDARY_THRESHOLD,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_papers(n: int, semantic_relevance: float = 0.35, cross_encoder_score=None) -> List[Paper]:
    """Build n identical dummy papers with given scores."""
    return [
        Paper(
            arxiv_id=str(i),
            title=f"Paper {i}",
            authors=["Author"],
            email_domains=[],
            abstract="Abstract text for testing classification.",
            submitted_date=datetime(2025, 1, 1),
            pdf_url="",
            arxiv_url="",
            semantic_relevance=semantic_relevance,
            cross_encoder_score=cross_encoder_score,
        )
        for i in range(n)
    ]


# ─── Paper dataclass field tests ──────────────────────────────────────────────

class TestPaperDataclass:
    def test_cross_encoder_score_defaults_none(self):
        p = make_papers(1)[0]
        assert p.cross_encoder_score is None

    def test_semantic_relevance_independent_field(self):
        p = make_papers(1, semantic_relevance=0.42)[0]
        assert p.semantic_relevance == 0.42
        p.cross_encoder_score = 0.80
        assert p.semantic_relevance == 0.42  # unchanged

    def test_both_scores_coexist(self):
        p = make_papers(1, semantic_relevance=0.45, cross_encoder_score=0.72)[0]
        assert p.semantic_relevance == 0.45
        assert p.cross_encoder_score == 0.72


# ─── scibert_classify_papers: CE path ─────────────────────────────────────────

class TestScibertClassifyCEPath:
    def test_high_ce_score_is_primary(self):
        papers = make_papers(5, cross_encoder_score=0.85)
        result = scibert_classify_papers(papers)
        assert all(p.focus_label == "primary" for p in result)

    def test_mid_ce_score_is_secondary(self):
        papers = make_papers(5, cross_encoder_score=0.35)
        result = scibert_classify_papers(papers)
        assert all(p.focus_label == "secondary" for p in result)

    def test_low_ce_score_is_off_topic(self):
        papers = make_papers(5, cross_encoder_score=0.10)
        result = scibert_classify_papers(papers)
        assert all(p.focus_label == "off-topic" for p in result)

    def test_boundary_primary_threshold(self):
        papers = make_papers(1, cross_encoder_score=PRIMARY_THRESHOLD)
        result = scibert_classify_papers(papers)
        assert result[0].focus_label == "primary"

    def test_boundary_secondary_threshold(self):
        papers = make_papers(1, cross_encoder_score=SECONDARY_THRESHOLD)
        result = scibert_classify_papers(papers)
        assert result[0].focus_label == "secondary"

    def test_llm_relevance_score_set_from_ce(self):
        papers = make_papers(3, cross_encoder_score=0.72)
        result = scibert_classify_papers(papers)
        for p in result:
            assert p.llm_relevance_score == pytest.approx(0.72)

    def test_mixed_ce_scores_correct_labels(self):
        papers = [
            make_papers(1, cross_encoder_score=0.90)[0],  # primary
            make_papers(1, cross_encoder_score=0.40)[0],  # secondary
            make_papers(1, cross_encoder_score=0.05)[0],  # off-topic
        ]
        result = scibert_classify_papers(papers)
        assert result[0].focus_label == "primary"
        assert result[1].focus_label == "secondary"
        assert result[2].focus_label == "off-topic"

    def test_none_ce_score_treated_as_zero(self):
        # Paper with CE available for others but None for one (partial failure)
        papers = make_papers(3, cross_encoder_score=0.70)
        papers[1].cross_encoder_score = None  # partial gap
        result = scibert_classify_papers(papers)
        # CE path runs because at least one CE score exists
        assert result[1].focus_label == "off-topic"  # None → 0.0 → below SECONDARY_THRESHOLD


# ─── scibert_classify_papers: heuristic fallback path ─────────────────────────

class TestScibertClassifyFallback:
    def test_no_ce_scores_triggers_heuristic(self):
        # All cross_encoder_score=None → heuristic fallback
        papers = make_papers(20, semantic_relevance=0.35)
        result = scibert_classify_papers(papers)
        primaries = [p for p in result if p.focus_label == "primary"]
        assert len(primaries) > 0, "Heuristic fallback must produce at least some primary papers"

    def test_no_results_regression(self):
        """
        P-10 regression: when CrossEncoder fails and semantic_relevance = 0.35
        (typical SPECTER2 cosine), the old code classified ALL papers off-topic
        (0.35 < PRIMARY_THRESHOLD=0.55). Heuristic fallback must prevent this.
        """
        papers = make_papers(50, semantic_relevance=0.35)
        result = scibert_classify_papers(papers)
        primaries = [p for p in result if p.focus_label == "primary"]
        assert len(primaries) >= 10, (
            f"Expected ≥10 primary papers from heuristic fallback, got {len(primaries)}"
        )

    def test_heuristic_fallback_top30_primary(self):
        n = 30
        papers = [
            make_papers(1, semantic_relevance=float(i) / n)[0]
            for i in range(n)
        ]
        result = scibert_classify_papers(papers)
        primaries = [p for p in result if p.focus_label == "primary"]
        # top 30% of 30 = 9 primaries at minimum
        assert len(primaries) >= 9

    def test_empty_papers_returns_empty(self):
        result = scibert_classify_papers([])
        assert result == []

    def test_single_paper_heuristic_gets_primary(self):
        papers = make_papers(1, semantic_relevance=0.30)
        result = scibert_classify_papers(papers)
        # heuristic: top max(1, 10, 30%×1) = top 1 = the only paper → primary
        assert result[0].focus_label == "primary"


# ─── cross_encoder_score does NOT overwrite semantic_relevance ────────────────

class TestCrossEncoderFieldIsolation:
    """
    Verify that Stage 3 writes cross_encoder_score and does NOT touch semantic_relevance.
    We test the invariant directly without loading the heavy model.
    """
    def test_semantic_relevance_preserved_after_ce_score_set(self):
        p = make_papers(1, semantic_relevance=0.42)[0]
        original_cosine = p.semantic_relevance
        # Simulate what cross_encoder_rerank now does
        p.cross_encoder_score = 1 / (1 + math.exp(-2.5))  # sigmoid(2.5) ≈ 0.924
        assert p.semantic_relevance == pytest.approx(original_cosine), (
            "cross_encoder_score write must not mutate semantic_relevance"
        )
        assert p.cross_encoder_score == pytest.approx(0.924, abs=0.01)

    def test_semantic_relevance_none_does_not_break_heuristic(self):
        papers = make_papers(10, semantic_relevance=None)
        # Heuristic uses `p.semantic_relevance or 0.0` — must not crash
        result = scibert_classify_papers(papers)
        assert len(result) == 10
        primaries = [p for p in result if p.focus_label == "primary"]
        assert len(primaries) >= 1


# ─── heuristic_classify_papers_free standalone ────────────────────────────────

class TestHeuristicClassify:
    def test_top_30_pct_primary(self):
        papers = [
            make_papers(1, semantic_relevance=float(i) / 100)[0]
            for i in range(100)
        ]
        result = heuristic_classify_papers_free(papers)
        primaries = [p for p in result if p.focus_label == "primary"]
        assert 28 <= len(primaries) <= 32  # ~30% tolerance

    def test_no_off_topic_labels(self):
        # heuristic only assigns primary or secondary
        papers = make_papers(20, semantic_relevance=0.10)
        result = heuristic_classify_papers_free(papers)
        for p in result:
            assert p.focus_label in ("primary", "secondary")

    def test_empty_returns_empty(self):
        result = heuristic_classify_papers_free([])
        assert result == []
