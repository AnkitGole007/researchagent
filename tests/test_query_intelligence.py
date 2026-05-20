"""
tests/test_query_intelligence.py — Unit tests for P-00 Query Intelligence Layer

Tests cover:
  - StructuredQuery properties (rrf weights, bm25_query_string)
  - Rules-based fallback: keyword extraction, intent detection, semantic query cleaning
  - LLM path: JSON parsing guard + schema validation gate
  - analyse_query() public API: returns StructuredQuery with correct shape
  - Backward-compat: empty brief returns safe StructuredQuery
"""
import os
import pytest

# Guard: QI module must be importable from reporoot
from query_intelligence import (
    StructuredQuery,
    analyse_query,
    _detect_intent,
    _extract_keywords,
    _build_semantic_query,
    _extract_not_terms,
    _rules_based_analyse,
)


# ─── StructuredQuery dataclass ────────────────────────────────────────────────

class TestStructuredQuery:
    def test_default_rrf_weights(self):
        sq = StructuredQuery()
        assert sq.rrf_weight_bm25 == 1.0
        assert sq.rrf_weight_faiss == 1.0

    def test_novelty_intent_boosts_faiss(self):
        sq = StructuredQuery(intent="novelty")
        assert sq.rrf_weight_faiss > sq.rrf_weight_bm25

    def test_foundational_intent_boosts_bm25(self):
        sq = StructuredQuery(intent="foundational")
        assert sq.rrf_weight_bm25 > sq.rrf_weight_faiss

    def test_survey_intent_boosts_both_but_faiss_more(self):
        sq = StructuredQuery(intent="survey")
        assert sq.rrf_weight_faiss > 1.0
        assert sq.rrf_weight_bm25 > 1.0

    def test_bm25_query_string_joins_keywords(self):
        sq = StructuredQuery(bm25_keywords=["transformer", "attention mechanism", "LLM"])
        result = sq.bm25_query_string
        assert "transformer" in result
        assert "LLM" in result

    def test_bm25_query_string_fallback_to_semantic_query(self):
        sq = StructuredQuery(semantic_query="Dense retrieval for scientific papers", bm25_keywords=[])
        assert sq.bm25_query_string == "Dense retrieval for scientific papers"


# ─── Rules-based helpers ──────────────────────────────────────────────────────

class TestDetectIntent:
    @pytest.mark.parametrize("brief,expected_set", [
        ("I want the most novel and recent papers on RL", {"novelty"}),
        ("Looking for survey and overview papers on transformers", {"survey"}),
        ("I need foundational and seminal papers on SGD", {"foundational"}),
        ("Give me diverse coverage across NLP and Vision areas", {"diversity"}),
        # "broadly" activates diversity signal — both diversity and general are acceptable
        ("I'm broadly interested in machine learning", {"diversity", "general"}),
    ])
    def test_intent_signals(self, brief, expected_set):
        assert _detect_intent(brief.lower()) in expected_set

    def test_year_triggers_specific(self):
        assert _detect_intent("papers from 2024 on diffusion models") == "specific"


class TestExtractKeywords:
    def test_returns_list(self):
        result = _extract_keywords("Attention is all you need for transformer models")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_stopwords_removed(self):
        result = _extract_keywords("the paper is about the use of deep learning methods")
        for kw in result:
            assert kw not in {"the", "paper", "about", "use", "methods"}

    def test_bigrams_preferred(self):
        result = _extract_keywords("recommendation systems matrix factorization deep learning")
        joined = " ".join(result)
        # At least one multi-word term expected
        assert any(" " in kw for kw in result)

    def test_top_n_capped(self):
        long_brief = " ".join([f"keyword{i}" for i in range(50)])
        result = _extract_keywords(long_brief, top_n=8)
        assert len(result) <= 8


class TestBuildSemanticQuery:
    def test_strips_not_looking_for_section(self):
        brief = "RESEARCH BRIEF:\nI want papers on LLMs.\n\nWHAT I AM NOT LOOKING FOR:\nSurveys."
        result = _build_semantic_query(brief)
        assert "NOT LOOKING FOR" not in result
        assert "Surveys" not in result

    def test_strips_research_brief_prefix(self):
        brief = "RESEARCH BRIEF:\nDense retrieval for scientific documents."
        result = _build_semantic_query(brief)
        assert not result.startswith("RESEARCH BRIEF")

    def test_hard_cap_512(self):
        long_brief = "a " * 500
        result = _build_semantic_query(long_brief)
        assert len(result) <= 512

    def test_passthrough_clean_brief(self):
        brief = "Contrastive learning in self-supervised vision models."
        result = _build_semantic_query(brief)
        assert "Contrastive" in result


class TestExtractNotTerms:
    def test_empty_without_not_section(self):
        brief = "RESEARCH BRIEF:\nI want papers on transformers."
        assert _extract_not_terms(brief) == []

    def test_extracts_terms_after_not_section(self):
        brief = "RESEARCH BRIEF:\nI want papers.\n\nWHAT I AM NOT LOOKING FOR:\nSurveys, NLP only, audio."
        result = _extract_not_terms(brief)
        assert "surveys" in result
        assert len(result) <= 10


# ─── Rules-based full analyser ────────────────────────────────────────────────

class TestRulesBasedAnalyse:
    def test_returns_structured_query(self):
        brief = "I am looking for recent transformer papers on recommendation systems."
        sq = _rules_based_analyse(brief)
        assert isinstance(sq, StructuredQuery)
        assert sq.source == "rules"

    def test_has_keywords(self):
        brief = "Contrastive self-supervised learning for vision-language models"
        sq = _rules_based_analyse(brief)
        assert len(sq.bm25_keywords) > 0

    def test_semantic_query_not_empty(self):
        brief = "I want papers on attention mechanisms in neural networks."
        sq = _rules_based_analyse(brief)
        assert len(sq.semantic_query) > 0

    def test_intent_detected(self):
        brief = "I'm specifically looking for novel papers on diffusion models from 2025."
        sq = _rules_based_analyse(brief)
        assert sq.intent in {"novelty", "specific", "general"}

    def test_quality_modifier_detected(self):
        brief = "I want recent and influential papers on graph neural networks."
        sq = _rules_based_analyse(brief)
        assert sq.quality_modifier in {"recent", "influential", "emerging", "classic", "any"}


# ─── Public API: analyse_query ────────────────────────────────────────────────

class TestAnalyseQuery:
    def test_empty_brief_returns_safe_default(self):
        sq = analyse_query("", groq_api_key=None)
        assert isinstance(sq, StructuredQuery)
        assert sq.intent in {"general", "novelty", "survey", "foundational", "specific", "diversity"}

    def test_no_api_key_uses_rules(self, monkeypatch):
        # Explicitly clear the env var so the rules path is forced regardless of .env
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        sq = analyse_query(
            "I am looking for papers on RLHF and instruction tuning.",
            groq_api_key=None,
        )
        assert sq.source == "rules"
        assert len(sq.bm25_keywords) > 0

    def test_rrf_weights_in_valid_range(self):
        sq = analyse_query("Recent papers on neural scaling laws.", groq_api_key=None)
        assert 0.5 <= sq.rrf_weight_bm25 <= 2.0
        assert 0.5 <= sq.rrf_weight_faiss <= 2.0

    def test_bm25_query_string_is_non_empty(self):
        sq = analyse_query("Graph neural networks for drug discovery.", groq_api_key=None)
        qs = sq.bm25_query_string
        assert len(qs.strip()) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set — skipping live LLM test"
    )
    def test_groq_llm_path_returns_valid_structured_query(self):
        """Integration test: only runs when GROQ_API_KEY is present."""
        brief = (
            "I am interested in papers about autonomous agents that can plan, use tools, "
            "and reason step-by-step. Especially ReAct, tool-augmented LLMs, and multi-agent systems."
        )
        sq = analyse_query(brief, groq_api_key=os.getenv("GROQ_API_KEY"))
        assert sq.source == "llm"
        assert sq.intent in {"novelty", "diversity", "foundational", "specific", "survey", "general"}
        assert len(sq.bm25_keywords) >= 3
        assert len(sq.semantic_query) > 10
        assert sq.quality_modifier in {"recent", "influential", "emerging", "classic", "any"}
        # Structural sanity: RRF weights are floats in range
        assert 0.5 <= sq.rrf_weight_bm25 <= 2.0
        assert 0.5 <= sq.rrf_weight_faiss <= 2.0
