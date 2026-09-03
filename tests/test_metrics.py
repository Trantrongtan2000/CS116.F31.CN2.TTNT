# -*- coding: utf-8 -*-
"""Behavior tests for mrc.metrics (Exact Match / token-F1)."""
from mrc import metrics


def test_normalize_strips_case_punctuation_and_whitespace():
    assert metrics.normalize_text("  Ngày 8, Tháng 6! ") == "ngày 8 tháng 6"


def test_exact_match_ignores_punctuation_and_case():
    assert metrics.exact_match("viện bảo tàng Louvre,", "viện bảo tàng Louvre") == 1


def test_exact_match_rejects_different_answer():
    assert metrics.exact_match("Hà Nội", "Sài Gòn") == 0


def test_f1_partial_overlap():
    # 2 shared tokens ("b","c"); pred=3 tokens, gt=3 tokens -> P=R=2/3 -> F1=2/3
    assert abs(metrics.f1("a b c", "b c d") - (2 / 3)) < 1e-9


def test_f1_no_overlap_is_zero():
    assert metrics.f1("x y", "a b") == 0.0


def test_f1_both_empty_is_perfect():
    # both "no answer" -> perfect (impossible question answered with empty)
    assert metrics.f1("", "") == 1.0


def test_f1_one_empty_is_zero():
    assert metrics.f1("something", "") == 0.0
    assert metrics.f1("", "something") == 0.0


def test_max_over_ground_truths_takes_best():
    assert metrics.metric_max_over_ground_truths(
        metrics.exact_match, "Hà Nội", ["Sài Gòn", "Hà Nội"]) == 1.0


def test_impossible_question_scored_correct_when_prediction_empty():
    # empty golds => correct iff prediction is empty
    assert metrics.metric_max_over_ground_truths(metrics.exact_match, "", []) == 1.0
    assert metrics.metric_max_over_ground_truths(metrics.exact_match, "guess", []) == 0.0


def test_evaluate_aggregates_percentages():
    preds = {"1": "Hà Nội", "2": "wrong"}
    refs = {"1": ["Hà Nội"], "2": ["Sài Gòn"]}
    out = metrics.evaluate(preds, refs)
    assert out["count"] == 2
    assert out["EM"] == 50.0        # 1 of 2 exact
    assert 0.0 <= out["F1"] <= 100.0


def test_evaluate_empty_is_zero_not_crash():
    assert metrics.evaluate({}, {}) == {"EM": 0.0, "F1": 0.0, "count": 0}


def test_per_item_scores_shape():
    preds = {"1": "Hà Nội"}
    refs = {"1": ["Hà Nội"]}
    scores = metrics.per_item_scores(preds, refs)
    assert scores["1"] == {"em": 1, "f1": 1.0}
