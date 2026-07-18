"""
tests/test_moneyball_weights.py

QIL quality_modifier -> Moneyball weight override (docs/qil-improvements-planner.md
B1, Job 2). Checks the precedence rule in resolve_moneyball_weights: a recognized
quality_modifier (influential/emerging/classic) wins outright; "any"/"recent" fall
through to moneyball_weights.json if present, else DEFAULT_MONEYBALL_WEIGHTS.
"""
from datetime import datetime, timedelta

from backend import pipeline_core as pc


def test_recognized_modifier_overrides_default():
    for modifier in ("influential", "emerging", "classic"):
        weights = pc.resolve_moneyball_weights(modifier)
        assert weights == pc.QUALITY_MONEYBALL_WEIGHTS[modifier]
        assert weights is not pc.DEFAULT_MONEYBALL_WEIGHTS


def test_any_and_recent_fall_back_to_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no moneyball_weights.json here
    assert pc.resolve_moneyball_weights("any") == pc.DEFAULT_MONEYBALL_WEIGHTS
    assert pc.resolve_moneyball_weights("recent") == pc.DEFAULT_MONEYBALL_WEIGHTS


def test_weight_dicts_have_matching_keys():
    for modifier, weights in pc.QUALITY_MONEYBALL_WEIGHTS.items():
        assert set(weights.keys()) == set(pc.DEFAULT_MONEYBALL_WEIGHTS.keys())


def test_recency_score_prefers_newer_papers():
    today_score = pc.compute_recency_score(datetime.now())
    old_score = pc.compute_recency_score(datetime.now() - timedelta(days=60))
    assert today_score > old_score
    assert old_score == 0.0
