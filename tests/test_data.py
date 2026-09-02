# -*- coding: utf-8 -*-
"""Behavior tests for mrc.data (parsing, leakage guard, tagging, subset)."""
import pytest

from mrc import data
from mrc.data import Example


def _squad(paragraphs):
    return {"version": "2.0", "data": [{"title": "T", "paragraphs": paragraphs}]}


def test_to_examples_parses_dict_of_lists_answers():
    squad = _squad([{
        "context": "Paris là thủ đô của Pháp.",
        "qas": [{"id": "q1", "question": "Thủ đô của Pháp là gì?",
                 "is_impossible": False,
                 "answers": {"text": ["Paris"], "answer_start": [0]}}],
    }])
    ex = data.to_examples(squad)
    assert len(ex) == 1
    assert ex[0].answers == ["Paris"]
    assert ex[0].has_answer is True


def test_to_examples_marks_impossible_and_reads_plausible():
    squad = _squad([{
        "context": "Paris là thủ đô của Pháp.",
        "qas": [{"id": "q2", "question": "Dân số Paris là bao nhiêu?",
                 "is_impossible": True,
                 "answers": {"text": [], "answer_start": []},
                 "plausible_answers": {"text": ["2 triệu"], "answer_start": [0]}}],
    }])
    ex = data.to_examples(squad)[0]
    assert ex.is_impossible is True
    assert ex.has_answer is False
    assert ex.plausible_answers == ["2 triệu"]


def test_empty_answer_list_is_treated_as_impossible():
    squad = _squad([{
        "context": "abc def ghi.",
        "qas": [{"id": "q3", "question": "?", "answers": {"text": [], "answer_start": []}}],
    }])
    assert data.to_examples(squad)[0].is_impossible is True


def test_assert_no_leakage_passes_when_disjoint():
    a = [Example("1", "", "ctx A", "q", ["x"], [0], False)]
    b = [Example("2", "", "ctx B", "q", ["y"], [0], False)]
    data.assert_no_leakage(a, b)  # must not raise


def test_assert_no_leakage_raises_on_shared_context():
    shared = "the same context text"
    a = [Example("1", "", shared, "q", ["x"], [0], False)]
    b = [Example("2", "", shared, "q", ["y"], [0], False)]
    with pytest.raises(AssertionError):
        data.assert_no_leakage(a, b)


def test_question_type_multihop_cue_is_multi_sentence():
    ctx = "Trời mưa. Đường trơn. Xe dừng lại."
    assert data.classify_question_type(ctx, "Tại sao xe dừng lại?", "Đường trơn", -1) == "multi-sentence"


def test_question_type_high_overlap_is_single_sentence():
    ctx = "Trường thành lập năm 2006 tại Thành phố Hồ Chí Minh."
    q = "Trường thành lập năm nào?"
    assert data.classify_question_type(ctx, q, "2006", ctx.find("2006")) == "single-sentence"


def test_subset_is_deterministic_with_seed():
    items = [Example(str(i), "", f"c{i}", "q", ["a"], [0], False) for i in range(20)]
    s1 = data.subset(items, 5, seed=42)
    s2 = data.subset(items, 5, seed=42)
    assert [e.id for e in s1] == [e.id for e in s2]
    assert len(s1) == 5


def test_subset_returns_all_when_n_exceeds_size():
    items = [Example("1", "", "c", "q", ["a"], [0], False)]
    assert data.subset(items, 10) is items


def test_references_dict_maps_id_to_gold_answers():
    items = [Example("q1", "", "c", "q", ["Paris"], [0], False)]
    assert data.references_dict(items) == {"q1": ["Paris"]}
