"""
tests/test_query_intelligence.py — Unit tests for P-00 Query Intelligence Layer

Tests cover:
  - StructuredQuery properties (rrf weights, bm25_query_string)
  - Rules-based fallback: keyword extraction, intent detection, semantic query cleaning
  - LLM path: JSON parsing guard + schema validation gate
  - analyse_query() public API: returns StructuredQuery with correct shape
  - YAKE keyword floor: bm25_keywords ≥ 5 after both paths
  - Multi-provider chain: source field values "llm_groq" | "llm_openrouter" | "rules"
  - Backward-compat: empty brief returns safe StructuredQuery
"""
import os
import pytest

from query_intelligence import (
    HardFilters,
    StructuredQuery,
    analyse_query,
    _detect_intent,
    _extract_keywords,
    _build_semantic_query,
    _extract_not_terms,
    _extract_date_range,
    _rules_based_analyse,
    _LLM_PROMPT_TEMPLATE,
    _YAKE_FLOOR,
)
import query_intelligence as qi
from datetime import datetime


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

    def test_source_default_is_rules(self):
        sq = StructuredQuery()
        assert sq.source == "rules"


# ─── Rules-based helpers ──────────────────────────────────────────────────────

class TestDetectIntent:
    @pytest.mark.parametrize("brief,expected_set", [
        ("I want the most novel and recent papers on RL", {"novelty"}),
        ("Looking for survey and overview papers on transformers", {"survey"}),
        ("I need foundational and seminal papers on SGD", {"foundational"}),
        ("Give me diverse coverage across NLP and Vision areas", {"diversity"}),
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
        joined = " ".join(result).lower()
        # These function words should not appear as standalone keywords
        for word in ("the", "about", "use"):
            assert word not in joined.split()

    def test_bigrams_or_ngrams_preferred(self):
        result = _extract_keywords("recommendation systems matrix factorization deep learning")
        # At least one multi-word term expected (bigram or trigram)
        assert any(" " in kw for kw in result)

    def test_top_n_capped(self):
        long_brief = " ".join([f"keyword{i}" for i in range(50)])
        result = _extract_keywords(long_brief, top_n=8)
        assert len(result) <= 8

    def test_scientific_phrases_extracted(self):
        brief = "contrastive learning self-supervised vision language models CLIP"
        result = _extract_keywords(brief, top_n=10)
        joined = " ".join(result).lower()
        # At least one scientifically meaningful term should appear
        assert any(term in joined for term in ["contrastive", "learning", "vision", "language", "clip"])


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

    def test_extracts_inline_exclusions(self):
        for brief, expected in [
            ("Papers on transformers. Exclude surveys.", "surveys"),
            ("Papers on transformers, omit benchmarks.", "benchmarks"),
            ("Diffusion models, not interested in image generation.", "image generation"),
            ("Retrieval work. No papers on speech recognition.", "speech recognition"),
        ]:
            assert expected in _extract_not_terms(brief), brief

    def test_inline_exclusion_stops_at_sentence_end(self):
        # The clause after the period is a positive request — it must survive.
        brief = "Papers on RAG. Exclude surveys. I also want vector database work."
        result = _extract_not_terms(brief)
        assert "surveys" in result
        assert not any("vector database" in t for t in result)

    def test_splits_and_joined_exclusions(self):
        result = _extract_not_terms("Papers on RAG. Exclude surveys and benchmarks.")
        assert "surveys" in result
        assert "benchmarks" in result


# ─── R5 / B5: typed HardFilters ───────────────────────────────────────────────

class TestHardFilters:
    def test_empty_filter_is_none(self):
        assert HardFilters().to_lancedb_filter() is None

    def test_year_clauses(self):
        assert HardFilters(year_from=2022).to_lancedb_filter() == "year >= 2022"
        assert HardFilters(year_to=2024).to_lancedb_filter() == "year <= 2024"
        assert (
            HardFilters(year_from=2022, year_to=2024).to_lancedb_filter()
            == "year >= 2022 AND year <= 2024"
        )

    def test_authors_and_venues_are_not_emitted(self):
        # B3/B4 apply these as LIKE with a result-count fallback — they must not
        # leak into the unconditional WHERE.
        hf = HardFilters(authors=["Hinton"], venues=["NeurIPS"])
        assert hf.to_lancedb_filter() is None

    def test_from_raw_handles_dict_date_range(self):
        hf = HardFilters.from_raw({"date_range": {"from": "2022-01-01", "to": "2024-12-31"}})
        assert (hf.year_from, hf.year_to) == (2022, 2024)

    def test_from_raw_handles_tuple_date_range(self):
        hf = HardFilters.from_raw({"date_range": ["2019-05-01", None]})
        assert (hf.year_from, hf.year_to) == (2019, None)

    def test_from_raw_survives_llm_garbage(self):
        # 8B models emit strings where lists belong, or the wrong type entirely.
        assert HardFilters.from_raw(None) == HardFilters()
        assert HardFilters.from_raw("not a dict") == HardFilters()
        assert HardFilters.from_raw({"authors": "Hinton"}).authors == ["Hinton"]
        assert HardFilters.from_raw({"date_range": "sometime"}).year_from is None

    def test_from_raw_lowercases_not_terms(self):
        # pipeline_core matches these against a lowercased title+abstract.
        assert HardFilters.from_raw({"not_terms": ["Surveys", " NLP "]}).not_terms == [
            "surveys", "nlp",
        ]

    def test_from_raw_is_idempotent(self):
        hf = HardFilters(not_terms=["surveys"])
        assert HardFilters.from_raw(hf) is hf

    def test_year_clause_cannot_carry_injected_text(self):
        hf = HardFilters.from_raw({"date_range": {"from": "2020; DROP TABLE papers"}})
        assert hf.to_lancedb_filter() == "year >= 2020"


# ─── R7 / B3+B4: author + venue LIKE clauses ──────────────────────────────────

class TestEntityFilter:
    def test_none_when_empty(self):
        assert HardFilters().to_entity_filter() is None

    def test_or_within_a_field_and_across_fields(self):
        assert (
            HardFilters(venues=["NeurIPS", "ICML"]).to_entity_filter()
            == "(venue LIKE '%NeurIPS%' OR venue LIKE '%ICML%')"
        )
        assert HardFilters(authors=["Hinton"], venues=["NeurIPS"]).to_entity_filter() == (
            "(authors LIKE '%Hinton%') AND (venue LIKE '%NeurIPS%')"
        )

    def test_injection_cannot_escape_the_string_literal(self):
        # Names reach a WHERE clause verbatim, so the trust boundary is a whitelist.
        clause = HardFilters(authors=["O'Brien'; DROP TABLE papers--"]).to_entity_filter()
        inner = clause[clause.index("'%") + 2:clause.rindex("%'")]
        assert "'" not in inner, "quote survived — literal could be terminated"
        assert ";" not in inner
        assert "--" not in inner, "SQL comment marker survived"

    def test_real_hyphenated_names_survive(self):
        assert "Smith-Jones" in HardFilters(authors=["Smith-Jones"]).to_entity_filter()

    def test_author_extraction_from_by_construction(self):
        assert qi._extract_authors("Papers by Geoffrey Hinton on backprop") == ["Geoffrey Hinton"]
        assert qi._extract_authors("work by Hinton and LeCun") == ["Hinton", "LeCun"]
        assert qi._extract_authors("authored by Bengio") == ["Bengio"]

    def test_author_extraction_ignores_non_names(self):
        assert qi._extract_authors("papers by 2020") == []
        assert qi._extract_authors("ranked by Relevance") == []
        assert qi._extract_authors("speed up training by using LoRA") == []
        assert qi._extract_authors("diffusion models for text generation") == []

    def test_name_that_sanitises_to_nothing_is_skipped(self):
        assert HardFilters(authors=["'''"]).to_entity_filter() is None
        assert HardFilters(authors=["'''"], venues=["ICML"]).to_entity_filter() == (
            "(venue LIKE '%ICML%')"
        )


# ─── R6 / B2: date range extraction ───────────────────────────────────────────

class TestExtractDateRange:
    def test_explicit_ranges(self):
        for brief in ["papers from 2020 to 2023", "work between 2020 and 2023", "2020-2023 results"]:
            assert _extract_date_range(brief) == (2020, 2023), brief

    def test_range_is_ordered_regardless_of_input_order(self):
        assert _extract_date_range("2023 to 2020") == (2020, 2023)

    def test_open_ended_bounds(self):
        assert _extract_date_range("papers since 2019") == (2019, None)
        assert _extract_date_range("anything after 2021") == (2021, None)
        assert _extract_date_range("work before 2015") == (None, 2015)
        assert _extract_date_range("published prior to 2018") == (None, 2018)

    def test_last_n_years_is_relative_to_now(self):
        year_from, year_to = _extract_date_range("papers from the last 3 years")
        assert year_from == datetime.now().year - 3
        assert year_to is None

    def test_bare_year_is_ignored(self):
        # "the 2017 Transformer paper" names a work — treating it as a bound would
        # silently narrow the corpus to a single year.
        assert _extract_date_range("the 2017 Transformer paper") == (None, None)
        assert _extract_date_range("no dates here at all") == (None, None)

    def test_rules_path_populates_hard_filters(self):
        sq = _rules_based_analyse("Diffusion model papers from 2021 to 2024.")
        assert (sq.hard_filters.year_from, sq.hard_filters.year_to) == (2021, 2024)
        assert sq.hard_filters.to_lancedb_filter() == "year >= 2021 AND year <= 2024"


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


# ─── YAKE floor ───────────────────────────────────────────────────────────────

class TestYakeFloor:
    def test_rules_path_meets_floor(self, monkeypatch):
        """Rules path reaches _YAKE_FLOOR keywords when the brief has that many
        extractable terms — YAKE is statistical, so a very short brief can't."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        brief = "Transformers attention mechanisms NLP tasks 2024."
        sq = analyse_query(brief, groq_api_key=None, openrouter_api_key=None)
        assert len(sq.bm25_keywords) >= _YAKE_FLOOR

    def test_floor_does_not_duplicate_existing_keywords(self, monkeypatch):
        """YAKE supplements without adding exact duplicates."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        brief = "Contrastive self-supervised learning for vision-language models like CLIP."
        sq = analyse_query(brief, groq_api_key=None, openrouter_api_key=None)
        lower_kws = [k.lower() for k in sq.bm25_keywords]
        assert len(lower_kws) == len(set(lower_kws)), "Duplicate keywords found after YAKE floor"


# ─── R8 / A1: synonym expansion ───────────────────────────────────────────────

class TestSynonymExpansion:
    def _fake_groq(self, keywords):
        """Stand in for a Groq response carrying `keywords` as bm25_keywords."""
        return lambda brief, api_key, model: (
            {
                "intent": "general",
                "semantic_query": "expanded statement",
                "bm25_keywords": keywords,
                "hard_filters": {"not_terms": [], "authors": [], "venues": []},
                "quality_modifier": "any",
            },
            None,
        )

    def test_prompt_asks_for_synonym_expansion(self):
        prompt = _LLM_PROMPT_TEMPLATE.format(brief="x")
        assert "synonym" in prompt.lower()
        assert "12-15" in prompt

    def test_expanded_keyword_list_survives_up_to_the_cap(self, monkeypatch):
        expanded = [
            "LoRA", "low-rank adaptation", "PEFT", "adapter", "parameter efficient",
            "QLoRA", "quantization", "fine-tuning", "instruction tuning", "LLM",
            "large language model", "transformer", "efficiency",
        ]
        monkeypatch.setattr(qi, "_call_groq_llm", self._fake_groq(expanded))
        sq = analyse_query("LoRA fine-tuning", groq_api_key="fake-key")
        assert sq.source == "llm_groq"
        assert sq.bm25_keywords == expanded, "13 expanded terms must pass through intact"

    def test_over_long_list_is_capped_not_dropped(self, monkeypatch):
        monkeypatch.setattr(qi, "_call_groq_llm", self._fake_groq([f"term{i}" for i in range(40)]))
        sq = analyse_query("anything", groq_api_key="fake-key")
        assert len(sq.bm25_keywords) == qi._MAX_LLM_KEYWORDS

    def test_thin_llm_result_is_topped_up_by_yake(self, monkeypatch):
        monkeypatch.setattr(qi, "_call_groq_llm", self._fake_groq(["transformers"]))
        sq = analyse_query(
            "Transformers attention mechanisms for NLP tasks and vision models.",
            groq_api_key="fake-key",
        )
        assert len(sq.bm25_keywords) >= _YAKE_FLOOR
        assert sq.bm25_keywords[0] == "transformers", "LLM terms keep priority over YAKE"

    def test_exclusion_regex_also_runs_on_the_llm_path(self, monkeypatch):
        """Live bug: the LLM returned not_terms=[] AND put "survey" in the keywords
        for a brief that said "Exclude surveys." — searching for the rejected thing."""
        monkeypatch.setattr(
            qi, "_call_groq_llm",
            self._fake_groq(["diffusion models", "survey", "generation"]),
        )
        sq = analyse_query(
            "Papers on diffusion models. Exclude surveys.", groq_api_key="fake-key"
        )
        assert "surveys" in sq.hard_filters.not_terms
        assert not any("survey" in kw.lower() for kw in sq.bm25_keywords), \
            "an excluded term must never remain a search term"

    def test_llm_date_range_is_dropped_when_the_brief_has_no_time_wording(self, monkeypatch):
        """Live bug: "diffusion models for text generation" came back with from=2020."""
        monkeypatch.setattr(qi, "_call_groq_llm", lambda brief, api_key, model: (
            {
                "intent": "general", "semantic_query": "s", "bm25_keywords": ["diffusion"],
                "hard_filters": {"date_range": {"from": "2020", "to": None}},
                "quality_modifier": "any",
            },
            None,
        ))
        sq = analyse_query("diffusion models for text generation", groq_api_key="fake-key")
        assert sq.hard_filters.year_from is None
        assert sq.hard_filters.to_lancedb_filter() is None

    def test_llm_date_range_survives_natural_language_time_wording(self, monkeypatch):
        """"post-ChatGPT era" has no parseable year — this is what the LLM path is for."""
        monkeypatch.setattr(qi, "_call_groq_llm", lambda brief, api_key, model: (
            {
                "intent": "general", "semantic_query": "s", "bm25_keywords": ["alignment"],
                "hard_filters": {"date_range": {"from": "2023", "to": None}},
                "quality_modifier": "any",
            },
            None,
        ))
        sq = analyse_query("alignment work from the post-ChatGPT era", groq_api_key="fake-key")
        assert sq.hard_filters.year_from == 2023

    def test_author_floor_fills_what_the_llm_missed(self, monkeypatch):
        """Live bug: the same "by Hinton" brief gave authors=["Hinton"] one run, [] the next."""
        monkeypatch.setattr(qi, "_call_groq_llm", lambda brief, api_key, model: (
            {
                "intent": "specific", "semantic_query": "s", "bm25_keywords": ["diffusion"],
                "hard_filters": {"authors": [], "venues": ["NeurIPS"]},
                "quality_modifier": "any",
            },
            None,
        ))
        sq = analyse_query(
            "Work by Hinton at NeurIPS on diffusion models.", groq_api_key="fake-key"
        )
        assert sq.hard_filters.authors == ["Hinton"]
        assert sq.hard_filters.venues == ["NeurIPS"]

    def test_ungrounded_entities_are_dropped(self, monkeypatch):
        """An invented author becomes a hard LIKE filter — it must not survive."""
        monkeypatch.setattr(qi, "_call_groq_llm", lambda brief, api_key, model: (
            {
                "intent": "general", "semantic_query": "s", "bm25_keywords": ["diffusion"],
                "hard_filters": {"authors": ["Schmidhuber"], "venues": ["ICML"]},
                "quality_modifier": "any",
            },
            None,
        ))
        sq = analyse_query("diffusion models for text generation", groq_api_key="fake-key")
        assert sq.hard_filters.authors == []
        assert sq.hard_filters.venues == []

    def test_quality_modifier_needs_support_in_the_brief(self, monkeypatch):
        """Live bug: a query with no recency wording came back "recent" -> year >= 2023."""
        def _fake(modifier):
            return lambda brief, api_key, model: (
                {
                    "intent": "general", "semantic_query": "s", "bm25_keywords": ["diffusion"],
                    "hard_filters": {}, "quality_modifier": modifier,
                },
                None,
            )

        monkeypatch.setattr(qi, "_call_groq_llm", _fake("recent"))
        assert analyse_query("diffusion models for text generation",
                             groq_api_key="k").quality_modifier == "any"

        monkeypatch.setattr(qi, "_call_groq_llm", _fake("influential"))
        assert analyse_query("diffusion models for text generation",
                             groq_api_key="k").quality_modifier == "any"

    def test_supported_quality_modifier_is_kept(self, monkeypatch):
        def _fake(modifier):
            return lambda brief, api_key, model: (
                {
                    "intent": "general", "semantic_query": "s", "bm25_keywords": ["diffusion"],
                    "hard_filters": {}, "quality_modifier": modifier,
                },
                None,
            )

        monkeypatch.setattr(qi, "_call_groq_llm", _fake("recent"))
        assert analyse_query("recent diffusion papers", groq_api_key="k").quality_modifier == "recent"

        monkeypatch.setattr(qi, "_call_groq_llm", _fake("influential"))
        assert analyse_query("highly cited diffusion papers",
                             groq_api_key="k").quality_modifier == "influential"

    def test_rules_path_is_unchanged_by_expansion(self, monkeypatch):
        """No regression: YAKE-only path still works when no key is present."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        sq = analyse_query(
            "Contrastive self-supervised learning for vision-language models.",
            groq_api_key=None,
            openrouter_api_key=None,
        )
        assert sq.source == "rules"
        assert len(sq.bm25_keywords) > 0


# ─── Public API: analyse_query ────────────────────────────────────────────────

class TestAnalyseQuery:
    def test_empty_brief_returns_safe_default(self):
        sq = analyse_query("", groq_api_key=None, openrouter_api_key=None)
        assert isinstance(sq, StructuredQuery)
        assert sq.intent in {"general", "novelty", "survey", "foundational", "specific", "diversity"}

    def test_no_api_key_uses_rules(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        sq = analyse_query(
            "I am looking for papers on RLHF and instruction tuning.",
            groq_api_key=None,
            openrouter_api_key=None,
        )
        assert sq.source == "rules"
        assert len(sq.bm25_keywords) > 0

    def test_rrf_weights_in_valid_range(self):
        sq = analyse_query("Recent papers on neural scaling laws.", groq_api_key=None, openrouter_api_key=None)
        assert 0.5 <= sq.rrf_weight_bm25 <= 2.0
        assert 0.5 <= sq.rrf_weight_faiss <= 2.0

    def test_bm25_query_string_is_non_empty(self):
        sq = analyse_query("Graph neural networks for drug discovery.", groq_api_key=None, openrouter_api_key=None)
        qs = sq.bm25_query_string
        assert len(qs.strip()) > 0

    def test_source_values_are_valid(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        sq = analyse_query("Diffusion models for image synthesis.", groq_api_key=None, openrouter_api_key=None)
        assert sq.source in {"llm_groq", "llm_openrouter", "rules"}

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set — skipping live Groq test"
    )
    def test_groq_llm_path_returns_valid_structured_query(self):
        """Integration test: only runs when GROQ_API_KEY is present."""
        brief = (
            "I am interested in papers about autonomous agents that can plan, use tools, "
            "and reason step-by-step. Especially ReAct, tool-augmented LLMs, and multi-agent systems."
        )
        sq = analyse_query(brief, groq_api_key=os.getenv("GROQ_API_KEY"), openrouter_api_key=None)
        assert sq.source == "llm_groq"
        assert sq.intent in {"novelty", "diversity", "foundational", "specific", "survey", "general"}
        assert len(sq.bm25_keywords) >= _YAKE_FLOOR
        assert len(sq.semantic_query) > 10
        assert sq.quality_modifier in {"recent", "influential", "emerging", "classic", "any"}
        assert 0.5 <= sq.rrf_weight_bm25 <= 2.0
        assert 0.5 <= sq.rrf_weight_faiss <= 2.0

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set — skipping live OpenRouter test"
    )
    def test_openrouter_path_returns_valid_structured_query(self):
        """Integration test: only runs when OPENROUTER_API_KEY is present."""
        brief = "Recent papers on diffusion models for image generation and editing."
        sq = analyse_query(
            brief,
            groq_api_key=None,       # force Groq skip
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        assert sq.source == "llm_openrouter"
        assert sq.intent in {"novelty", "diversity", "foundational", "specific", "survey", "general"}
        assert len(sq.bm25_keywords) >= _YAKE_FLOOR
        assert len(sq.semantic_query) > 10
