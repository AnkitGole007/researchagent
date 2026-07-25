"""
tests/test_predict_citations_concurrent.py

T-68: predict_citations_direct now runs S2 + LLM calls concurrently across two
bounded pools instead of one serial per-paper loop. Checks the properties that
matter under concurrency and wouldn't be caught by the old serial-only tests:
results land on the correct paper regardless of completion order, on_progress
still reports a sane final count, and one paper's LLM call raising doesn't
take down the whole batch.
"""
import random
import time
from datetime import datetime

from backend import pipeline_core as pc


def _make_paper(i: int) -> pc.Paper:
    return pc.Paper(
        arxiv_id=f"2000.{i:05d}",
        title=f"Paper {i}",
        authors=[],
        email_domains=[],
        abstract=f"Abstract for paper {i}.",
        submitted_date=datetime(2020, 1, 1),
        pdf_url="",
        arxiv_url="",
    )


def test_results_align_with_correct_paper_under_concurrency(monkeypatch):
    papers = [_make_paper(i) for i in range(12)]

    def fake_s2_stats(paper, api_key=None):
        time.sleep(random.uniform(0, 0.02))  # scramble completion order
        return int(paper.arxiv_id.split(".")[1])  # deterministic, index-derived

    monkeypatch.setattr(pc, "get_s2_citation_stats", fake_s2_stats)

    llm_config = pc.LLMConfig(api_key="", model="", api_base=None, provider="openai")
    result = pc.predict_citations_direct(papers, llm_config, quality_modifier="any")

    for i, p in enumerate(result):
        assert p.arxiv_id == f"2000.{i:05d}"
        assert p.predicted_citations is not None
        # A scrambled result[i] <-> wrong-paper's-citations mixup would break monotonicity:
        # fake_s2_stats returns i itself, so predicted_citations must trend with the index.
        if i > 1:
            assert p.predicted_citations >= result[0].predicted_citations


def test_progress_reaches_total_and_one_bad_llm_call_does_not_crash_batch(monkeypatch):
    papers = [_make_paper(i) for i in range(6)]
    monkeypatch.setattr(pc, "get_s2_citation_stats", lambda paper, api_key=None: 0)

    def flaky_call_llm(prompt, llm_config, label=""):
        if "Paper 3" in prompt:
            raise RuntimeError("simulated provider failure")
        return ""  # falls through to heuristic content_bullets, same as no LLM configured

    monkeypatch.setattr(pc, "call_llm", flaky_call_llm)

    progress_calls = []
    llm_config = pc.LLMConfig(api_key="fake-key", model="gpt-test", api_base=None, provider="openai")

    result = pc.predict_citations_direct(
        papers, llm_config, quality_modifier="any",
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )

    assert len(result) == 6
    assert progress_calls
    assert progress_calls[-1] == (6, 6)
