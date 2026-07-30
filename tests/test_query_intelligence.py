"""
tests/test_query_intelligence.py — Unit tests for P-00 Query Intelligence Layer

Tests cover:
  - StructuredQuery properties (rrf weights, bm25_query_string)
  - Rules-based fallback: keyword extraction, intent detection, semantic query cleaning
  - LLM path: JSON parsing guard + schema validation gate
  - analyse_query() public API: returns StructuredQuery with correct shape
  - YAKE keyword floor: bm25_keywords ≥ 5 after both paths
  - Multi-provider chain: source field values "llm_groq" | "llm_openrouter" | "llm_ollama" | "rules"
  - Backward-compat: empty brief returns safe StructuredQuery
"""
import os
import pytest

from query_intelligence import (
    Criterion,
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
    """_detect_intent takes the raw brief now, not a pre-lowered string — it does
    its own case-insensitive matching internally (word-boundary patterns are
    compiled with re.IGNORECASE) and needs original casing for _extract_authors."""

    @pytest.mark.parametrize("brief,expected_set", [
        ("I want the most novel and recent papers on RL", {"novelty"}),
        ("Looking for survey and overview papers on transformers", {"survey"}),
        ("I need foundational and seminal papers on SGD", {"foundational"}),
        ("Give me diverse coverage across NLP and Vision areas", {"diversity"}),
        ("I'm broadly interested in machine learning", {"diversity", "general"}),
    ])
    def test_intent_signals(self, brief, expected_set):
        assert _detect_intent(brief) in expected_set

    def test_year_triggers_specific(self):
        assert _detect_intent("papers from 2024 on diffusion models") == "specific"

    def test_named_author_triggers_specific(self):
        # Was dead code before this fix: the old regex required an uppercase
        # letter class against an always-lowercased string, so it could never
        # match. Now reuses _extract_authors instead of a second, broken check.
        assert _detect_intent("papers by Geoffrey Hinton on backprop") == "specific"

    def test_plural_signals_still_match(self):
        # Word-boundary matching must not exclude simple plurals — "survey" alone
        # has no boundary before the "s" in "surveys".
        assert _detect_intent("recent surveys on retrieval augmented generation") == "survey"

    @pytest.mark.parametrize("brief", [
        "a peer-reviewed paper on transformers",       # "review" must not match inside "reviewed"
        "machine learning for renewable energy forecasting",  # "new" must not match inside "renewable"
        "newton method optimization",                  # "new" must not match inside "newton"
    ])
    def test_substring_false_positives_fixed(self, brief):
        # Previously unbounded `s in brief_lower` matched these as real signals.
        assert _detect_intent(brief) == "general"

    def test_ordering_favours_specific_signal_over_novelty(self):
        # Previously: novelty was checked first and its bare "recent"/"new"
        # signals won every tie, even against an explicit "overview" match.
        assert _detect_intent(
            "give me a broad overview of approaches to reduce LLM hallucination"
        ) == "survey"

    @pytest.mark.parametrize("brief", [
        "papers on GANs and diffusion models, exclude surveys and tutorials",
        "RESEARCH BRIEF:\nI want papers on X.\n\nWHAT I AM NOT LOOKING FOR:\nSurveys",
    ])
    def test_negated_signal_does_not_flip_intent(self, brief):
        # Previously: "survey" inside the excluded clause flipped intent to
        # "survey" for a brief explicitly rejecting surveys.
        assert _detect_intent(brief) != "survey"


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

    def test_research_brief_wrapper_does_not_leak_into_keywords(self):
        """Live bug: the real caller (backend/pipeline_core.py's build_query_brief)
        wraps every brief as "RESEARCH BRIEF:\\n{text}" before it reaches QIL at all.
        n-gram extraction is sensitive to word adjacency across that boundary in a
        way substring matching isn't — bigram extraction on the raw wrapped text
        produced the literal keyword "brief: diffusion", the tail of "BRIEF:" paired
        with the first real word after it."""
        wrapped = "RESEARCH BRIEF:\ndiffusion models for text generation"
        result = _extract_keywords(wrapped, top_n=16)
        assert not any(":" in kw for kw in result)
        assert not any(kw.lower().startswith("brief") for kw in result)
        assert "diffusion models" in result

    def test_not_looking_for_section_does_not_leak_into_keywords(self):
        """The excluded topic must not contribute search keywords either — same
        principle as _apply_not_terms_floor, but for the rules/YAKE extraction path."""
        wrapped = (
            "RESEARCH BRIEF:\ndiffusion models for text generation\n\n"
            "WHAT I AM NOT LOOKING FOR:\nreinforcement learning approaches"
        )
        result = _extract_keywords(wrapped, top_n=16)
        assert not any("reinforcement" in kw.lower() for kw in result)

    def test_trailing_punctuation_does_not_leak_into_bigrams(self):
        """Live bug, twice, same root cause, two different punctuation marks:
        a word immediately followed by the user's own punctuation ("systems:",
        "architectures,") carried that punctuation into the joined bigram,
        since the bigram step split on whitespace only and never stripped it."""
        brief = (
            "papers about recommendation systems: for example, new model "
            "architectures, training strategies, evaluation methods."
        )
        result = _extract_keywords(brief, top_n=16)
        assert not any(c in kw for kw in result for c in ":,;.")
        assert "recommendation systems" in result
        assert "model architectures" in result

    def test_hyphenated_words_survive_punctuation_stripping(self):
        """The fix must strip leading/trailing punctuation without breaking
        internal hyphens — "self-supervised" is one word, not two fragments."""
        result = _extract_keywords("self-supervised learning for vision tasks.", top_n=10)
        assert any("self-supervised" in kw for kw in result)


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


# ─── QIL v3 Stage 2: criteria decomposition ───────────────────────────────────

class TestCriterion:
    def test_valid_criterion_parses(self):
        c = Criterion.from_raw({"name": "X", "definition": "Y", "strength": "must"})
        assert c == Criterion(name="X", definition="Y", strength="must")

    def test_missing_definition_is_dropped_not_defaulted(self):
        # A criterion with an empty definition is worse than no criterion at all.
        assert Criterion.from_raw({"name": "X"}) is None
        assert Criterion.from_raw({"definition": "Y"}) is None
        assert Criterion.from_raw({"name": "", "definition": "Y"}) is None

    def test_non_dict_input_returns_none(self):
        assert Criterion.from_raw("not a dict") is None
        assert Criterion.from_raw(None) is None
        assert Criterion.from_raw(["X", "Y"]) is None

    def test_invalid_strength_defaults_to_should(self):
        c = Criterion.from_raw({"name": "X", "definition": "Y", "strength": "garbage"})
        assert c.strength == "should"

    def test_default_strength_is_should(self):
        c = Criterion.from_raw({"name": "X", "definition": "Y"})
        assert c.strength == "should"


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
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        brief = "Transformers attention mechanisms NLP tasks 2024."
        sq = analyse_query(brief, groq_api_key=None, openrouter_api_key=None)
        assert len(sq.bm25_keywords) >= _YAKE_FLOOR

    def test_floor_does_not_duplicate_existing_keywords(self, monkeypatch):
        """YAKE supplements without adding exact duplicates."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
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
        assert "8-12" in prompt

    def test_prompt_has_no_worked_example_tokens_to_anchor_on(self):
        """
        Live bug: the prompt used to teach synonym expansion with a worked example
        ("LoRA" -> "low-rank adaptation"/"PEFT"/"adapter"; "RLHF" -> "reward
        model"/"human feedback"/"preference data"), and the 8B model echoed those
        literal tokens into completely unrelated queries regardless of topic —
        confirmed in 6/20 live-audit briefs, none about fine-tuning or RLHF.
        The fix removes the worked example entirely, so there's nothing to leak.
        """
        prompt = _LLM_PROMPT_TEMPLATE.format(brief="x")
        for leaked in (
            "low-rank adaptation", "PEFT", "adapter", "reward model",
            "human feedback", "preference data",
        ):
            assert leaked.lower() not in prompt.lower(), (
                f"{leaked!r} is a former leaked example token — must not reappear in the prompt"
            )

    def test_expanded_keyword_list_survives_up_to_the_cap(self, monkeypatch):
        expanded = [
            "LoRA", "low-rank adaptation", "PEFT", "adapter", "parameter efficient",
            "QLoRA", "quantization", "fine-tuning", "instruction tuning", "LLM",
            "large language model", "transformer",
        ]
        assert len(expanded) == qi._MAX_LLM_KEYWORDS, "fixture must exactly fill the cap"
        monkeypatch.setattr(qi, "_call_groq_llm", self._fake_groq(expanded))
        sq = analyse_query("LoRA fine-tuning", groq_api_key="fake-key")
        assert sq.source == "llm_groq"
        assert sq.bm25_keywords == expanded, "terms at exactly the cap must pass through intact"

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
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        sq = analyse_query(
            "Contrastive self-supervised learning for vision-language models.",
            groq_api_key=None,
            openrouter_api_key=None,
        )
        assert sq.source == "rules"
        assert len(sq.bm25_keywords) > 0


# ─── QIL Auditor Stage A: intent-bias and acronym-hallucination floors ────────

def _fake_llm(intent="general", keywords=None, semantic_query="s", criteria=None):
    """Stand in for a Groq response with controlled intent/keywords/semantic_query/criteria."""
    return lambda brief, api_key, model: (
        {
            "intent": intent,
            "semantic_query": semantic_query,
            "bm25_keywords": keywords or ["placeholder"],
            "hard_filters": {"not_terms": [], "authors": [], "venues": []},
            "quality_modifier": "any",
            "criteria": criteria if criteria is not None else [],
        },
        None,
    )


class TestApplyIntentFloor:
    def test_overrides_when_rules_signal_is_unambiguous(self, monkeypatch):
        """Live bug: LLM said novelty for a brief that plainly asks for an overview."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(intent="novelty"))
        sq = analyse_query(
            "give me a broad overview of approaches to reduce LLM hallucination",
            groq_api_key="fake-key",
        )
        assert sq.intent == "survey"

    def test_does_not_override_when_rules_path_also_has_no_signal(self, monkeypatch):
        """If the rules path itself would say "general", there's no stronger ground
        truth to correct to — the LLM's answer must stand."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(intent="novelty"))
        sq = analyse_query("some vague research topic with no clear signal", groq_api_key="fake-key")
        assert sq.intent == "novelty"

    def test_no_change_when_llm_and_rules_agree(self, monkeypatch):
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(intent="survey"))
        sq = analyse_query("a survey of transformer architectures", groq_api_key="fake-key")
        assert sq.intent == "survey"


class TestApplyAcronymFloor:
    def test_corrects_dpo_and_kto_live_regression(self, monkeypatch):
        """Exact live bug: DPO/KTO expanded to fabricated full forms."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            keywords=["RLHF", "reward model", "DPO", "Differentially Private Optimization",
                      "KTO", "Knowledge Transfer Optimization", "alignment"],
            semantic_query="Comparing RLHF, DPO, and KTO for alignment.",
        ))
        sq = analyse_query("RLHF vs DPO vs KTO for alignment", groq_api_key="fake-key")
        assert "Direct Preference Optimization" in sq.bm25_keywords
        assert "Kahneman-Tversky Optimization" in sq.bm25_keywords
        assert "Differentially Private Optimization" not in sq.bm25_keywords
        assert "Knowledge Transfer Optimization" not in sq.bm25_keywords

    def test_related_keyword_is_not_mistaken_for_an_expansion_attempt(self, monkeypatch):
        """RLHF's real neighbours here are related concepts, not expansion attempts —
        their initials (R, H) don't correspond to RLHF's letters, so the letter gate
        must leave them untouched rather than "verifying" them against the table."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            keywords=["RLHF", "reward model", "human feedback", "preference data"],
        ))
        sq = analyse_query("papers on RLHF", groq_api_key="fake-key")
        assert "reward model" in sq.bm25_keywords
        assert "human feedback" in sq.bm25_keywords

    def test_already_correct_expansion_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            keywords=["DPO", "Direct Preference Optimization"],
        ))
        sq = analyse_query("papers on DPO", groq_api_key="fake-key")
        # Only the first two are asserted exactly — YAKE tops the rest up to
        # _YAKE_FLOOR since the fake LLM supplied just 2 keywords.
        assert sq.bm25_keywords[:2] == ["DPO", "Direct Preference Optimization"]

    def test_unlisted_acronym_is_left_untouched(self, monkeypatch):
        """Stage A has no escalation call yet — an unlisted acronym can't be
        verified, so it must not be dropped or altered."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            keywords=["XYZ", "Some Invented Full Form"],
        ))
        sq = analyse_query("papers on XYZ", groq_api_key="fake-key")
        assert sq.bm25_keywords[:2] == ["XYZ", "Some Invented Full Form"]

    def test_never_overwrites_the_brief_s_own_words(self, monkeypatch):
        """A differential-privacy researcher's own phrase must not be inverted just
        because it happens to letter-match DPO's initials."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            keywords=["DPO", "Differentially Private Optimization"],
        ))
        sq = analyse_query(
            "DPO (Differentially Private Optimization) for federated training",
            groq_api_key="fake-key",
        )
        assert "Differentially Private Optimization" in sq.bm25_keywords

    def test_semantic_query_gets_the_same_correction(self, monkeypatch):
        """semantic_query drives the SPECTER2 vector-search arm — a floor that only
        touches bm25_keywords leaves the poisoned phrase driving half of retrieval.

        Brief is deliberately long (>_VERBATIM_QUERY_MAX_CHARS) so the v3 Stage 1
        verbatim-query floor doesn't also fire and mask this floor's own effect —
        for a short brief the verbatim floor wins outright, which is correct
        (tested separately), but isn't what this test is checking."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            keywords=["DPO", "Differentially Private Optimization"],
            semantic_query="Recent work on DPO (Differentially Private Optimization).",
        ))
        brief = (
            "A detailed survey of preference optimization methods for aligning large "
            "language models with human feedback, focusing specifically on DPO and "
            "its variants, compared against reinforcement-learning-based approaches."
        )
        assert len(brief) > qi._VERBATIM_QUERY_MAX_CHARS
        sq = analyse_query(brief, groq_api_key="fake-key")
        assert "Direct Preference Optimization" in sq.semantic_query
        assert "Differentially Private Optimization" not in sq.semantic_query

    def test_acronym_table_entries_pass_their_own_gate(self):
        """The one runnable check this guard needs: every seed entry must be
        verifiable against its own key, or it's a permanent silent no-op."""
        for acronym, canonical in qi._ACRONYM_EXPANSIONS.items():
            assert qi._acronym_letters_match(acronym, canonical), (
                f"{acronym!r} -> {canonical!r} does not pass its own letter gate"
            )


# ─── QIL v3 Stage 1: term-quality floor and verbatim-query floor ──────────────

class TestApplyTermQualityFloor:
    def test_strips_hypernyms_and_intent_leak_words(self, monkeypatch):
        """Live bug: a query about diffusion models returned "artificial
        intelligence", "machine learning", "deep learning", "novel", "recent" as
        keywords — zero discriminative power in an all-AI 385K-paper corpus."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(keywords=[
            "diffusion model", "text generation", "artificial intelligence",
            "machine learning", "deep learning", "novel", "recent", "neural networks",
        ]))
        sq = analyse_query("diffusion models for text generation", groq_api_key="fake-key")
        assert "diffusion model" in sq.bm25_keywords
        assert "text generation" in sq.bm25_keywords
        for blacklisted in (
            "artificial intelligence", "machine learning", "deep learning",
            "novel", "recent", "neural networks",
        ):
            assert blacklisted not in sq.bm25_keywords

    def test_strips_prompt_echo_artifacts(self, monkeypatch):
        """Live bug: the model reproducibly echoed the prompt's own Brief:\"\"\"...\"\"\"
        wrapper as a literal keyword, "brief: diffusion" — same anchoring failure
        mode as R8's leaked worked-example tokens, different part of the prompt."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            keywords=["diffusion model", "brief: diffusion", "text: generation"],
        ))
        sq = analyse_query("diffusion models for text generation", groq_api_key="fake-key")
        assert "diffusion model" in sq.bm25_keywords
        assert not any(":" in kw for kw in sq.bm25_keywords)
        assert not any(kw.lower().startswith("brief") for kw in sq.bm25_keywords)

    def test_whole_term_match_only_compound_survives(self, monkeypatch):
        """"model" alone is generic; "diffusion model" is specific — a substring
        match would wrongly strip the compound too."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            keywords=["model", "diffusion model", "generative model"],
        ))
        sq = analyse_query("diffusion models", groq_api_key="fake-key")
        assert "model" not in sq.bm25_keywords
        assert "diffusion model" in sq.bm25_keywords
        assert "generative model" in sq.bm25_keywords

    def test_yake_top_up_also_respects_the_blacklist(self, monkeypatch):
        """The floor runs before the YAKE top-up — if YAKE's own candidates
        weren't also filtered, a stripped hypernym could just be re-added."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(keywords=["diffusion model"]))
        sq = analyse_query(
            "diffusion model research on novel deep learning methods for text generation",
            groq_api_key="fake-key",
        )
        for blacklisted in ("research", "novel", "deep learning", "methods"):
            assert blacklisted not in sq.bm25_keywords

    def test_yake_top_up_also_rejects_prompt_echo_artifacts(self, monkeypatch):
        """Live bug: "recommendation systems:" (trailing colon from the user's
        own punctuation, picked up by YAKE's bigram fallback) survived because
        the top-up loop only checked the hypernym blacklist, not the same
        colon/"brief"-prefix check the term-quality floor already applied."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(keywords=["recommendation"]))
        sq = analyse_query(
            "papers about recommendation systems: for example, new model architectures",
            groq_api_key="fake-key",
        )
        assert not any(":" in kw for kw in sq.bm25_keywords)

    def test_blacklist_entries_are_lowercase_and_singular_plural_paired(self):
        """Regression guard for the live gap this floor shipped with: "neural
        network" was blacklisted but "neural networks" wasn't, so the plural
        silently survived. Every countable-noun entry needs both forms."""
        # value = actual plural form ("approach" pluralizes to "approaches", not "approachs")
        countable_singulars = {
            "method": "methods", "approach": "approaches", "technique": "techniques",
            "algorithm": "algorithms", "framework": "frameworks", "system": "systems",
            "paper": "papers", "result": "results", "model": "models",
        }
        for singular, plural in countable_singulars.items():
            assert singular in qi._HYPERNYM_TERMS, f"singular {singular!r} missing from blacklist"
            assert plural in qi._HYPERNYM_TERMS, f"plural {plural!r} missing from blacklist"
        # "neural network" is a two-word compound, so the same pairing check
        # applies to the phrase, not a bare "network"/"networks" entry.
        assert "neural network" in qi._HYPERNYM_TERMS
        assert "neural networks" in qi._HYPERNYM_TERMS


class TestApplyVerbatimQueryFloor:
    def test_short_brief_gets_the_brief_not_a_paraphrase(self, monkeypatch):
        """Live A/B: verbatim scored 83/100 on-topic vs. 3/100 for the drifted
        paraphrase of the same short brief (docs/PLAN.md)."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(
            semantic_query="Recent advancements in diffusion models have led to "
                          "improved performance and efficiency.",
        ))
        brief = "diffusion models for text generation"
        assert len(brief) <= qi._VERBATIM_QUERY_MAX_CHARS
        sq = analyse_query(brief, groq_api_key="fake-key")
        assert sq.semantic_query == brief

    def test_long_brief_keeps_the_llm_paraphrase(self, monkeypatch):
        paraphrase = "A synthesis of self-supervised vision techniques for autonomous driving perception."
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(semantic_query=paraphrase))
        brief = (
            "Work by Yoshua Bengio and Yann LeCun published at ICML or ICLR since 2018 "
            "on self-supervised learning for computer vision applications in autonomous "
            "driving systems, with particular attention to perception robustness."
        )
        assert len(brief) > qi._VERBATIM_QUERY_MAX_CHARS
        sq = analyse_query(brief, groq_api_key="fake-key")
        assert sq.semantic_query == paraphrase

    def test_verbatim_query_is_cleaned_not_raw(self, monkeypatch):
        """Reuses _build_semantic_query — strips the NOT-section, not just a
        length-gated passthrough of the raw brief."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm())
        brief = "RESEARCH BRIEF:\nDiffusion models.\n\nWHAT I AM NOT LOOKING FOR:\nSurveys."
        assert len(brief) <= qi._VERBATIM_QUERY_MAX_CHARS
        sq = analyse_query(brief, groq_api_key="fake-key")
        assert sq.semantic_query == "Diffusion models."


# ─── Public API: analyse_query ────────────────────────────────────────────────

class TestAnalyseQuery:
    def test_empty_brief_returns_safe_default(self):
        sq = analyse_query("", groq_api_key=None, openrouter_api_key=None)
        assert isinstance(sq, StructuredQuery)
        assert sq.intent in {"general", "novelty", "survey", "foundational", "specific", "diversity"}

    def test_no_api_key_uses_rules(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
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
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        sq = analyse_query("Diffusion models for image synthesis.", groq_api_key=None, openrouter_api_key=None)
        assert sq.source in {"llm_groq", "llm_openrouter", "llm_ollama", "rules"}

    def test_escalates_to_ollama_on_openrouter_rate_limit(self, monkeypatch):
        """Ollama Cloud is an alternate fallback tier, one step past OpenRouter —
        only reached once OpenRouter itself is degraded, same relationship
        OpenRouter has with Groq."""
        monkeypatch.setattr(qi, "_call_groq_llm", lambda brief, api_key, model: (None, "rate_limit"))
        monkeypatch.setattr(qi, "_call_openrouter_llm", lambda brief, api_key, model: (None, "rate_limit"))
        monkeypatch.setattr(qi, "_call_ollama_llm", _fake_llm(keywords=["diffusion models"]))
        sq = analyse_query(
            "diffusion models for text generation",
            groq_api_key="fake-key",
            openrouter_api_key="fake-key",
            ollama_api_key="fake-key",
        )
        assert sq.source == "llm_ollama"

    def test_openrouter_parse_error_skips_ollama_falls_to_rules(self, monkeypatch):
        """Mirrors Groq's own parse-error handling: a parse failure means 'this
        model's output was bad', not 'the provider is degraded' — escalating
        further wouldn't help, so it should go straight to rules, same as a
        Groq parse error skips OpenRouter entirely."""
        def _boom(brief, api_key, model):
            raise AssertionError("Ollama should not be called on an OpenRouter parse error")

        monkeypatch.setattr(qi, "_call_groq_llm", lambda brief, api_key, model: (None, "rate_limit"))
        monkeypatch.setattr(qi, "_call_openrouter_llm", lambda brief, api_key, model: (None, "parse"))
        monkeypatch.setattr(qi, "_call_ollama_llm", _boom)
        sq = analyse_query(
            "diffusion models for text generation",
            groq_api_key="fake-key",
            openrouter_api_key="fake-key",
            ollama_api_key="fake-key",
        )
        assert sq.source == "rules"

    def test_ollama_failure_falls_to_rules(self, monkeypatch):
        monkeypatch.setattr(qi, "_call_groq_llm", lambda brief, api_key, model: (None, "rate_limit"))
        monkeypatch.setattr(qi, "_call_openrouter_llm", lambda brief, api_key, model: (None, "rate_limit"))
        monkeypatch.setattr(qi, "_call_ollama_llm", lambda brief, api_key, model: (None, "other"))
        sq = analyse_query(
            "diffusion models for text generation",
            groq_api_key="fake-key",
            openrouter_api_key="fake-key",
            ollama_api_key="fake-key",
        )
        assert sq.source == "rules"

    def test_rules_path_never_produces_criteria(self, monkeypatch):
        """Criteria decomposition needs genuine language understanding a
        deterministic fallback can't provide — no criteria beats fake ones."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        sq = analyse_query("diffusion models", groq_api_key=None, openrouter_api_key=None)
        assert sq.source == "rules"
        assert sq.criteria == []

    def test_llm_criteria_pass_through(self, monkeypatch):
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(criteria=[
            {"name": "Mechanism", "definition": "Proposes a new sparse attention mechanism.", "strength": "must"},
            {"name": "Benchmark", "definition": "Includes empirical results on long-context tasks.", "strength": "should"},
        ]))
        sq = analyse_query("sparse attention for long context", groq_api_key="fake-key")
        assert sq.criteria == [
            Criterion(name="Mechanism", definition="Proposes a new sparse attention mechanism.", strength="must"),
            Criterion(name="Benchmark", definition="Includes empirical results on long-context tasks.", strength="should"),
        ]

    def test_criteria_capped_at_five(self, monkeypatch):
        raw_criteria = [
            {"name": f"C{i}", "definition": f"Definition {i}.", "strength": "should"} for i in range(8)
        ]
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(criteria=raw_criteria))
        sq = analyse_query("anything", groq_api_key="fake-key")
        assert len(sq.criteria) == 5

    def test_malformed_criteria_items_are_dropped_not_fatal(self, monkeypatch):
        """One bad item in the list must not sink the whole response."""
        monkeypatch.setattr(qi, "_call_groq_llm", _fake_llm(criteria=[
            {"name": "Valid", "definition": "A real criterion.", "strength": "must"},
            {"name": "NoDefinition"},
            "not even a dict",
            None,
        ]))
        sq = analyse_query("anything", groq_api_key="fake-key")
        assert sq.criteria == [Criterion(name="Valid", definition="A real criterion.", strength="must")]

    def test_criteria_field_absent_from_llm_response_is_safe(self, monkeypatch):
        """An 8B model may omit the key entirely — must not raise."""
        monkeypatch.setattr(qi, "_call_groq_llm", lambda brief, api_key, model: (
            {
                "intent": "general", "semantic_query": "s", "bm25_keywords": ["x"],
                "hard_filters": {}, "quality_modifier": "any",
            },
            None,
        ))
        sq = analyse_query("anything", groq_api_key="fake-key")
        assert sq.criteria == []

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
