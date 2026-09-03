# -*- coding: utf-8 -*-
"""Exact Match (EM) and token-level F1 for extractive MRC.

Follows the SQuAD scoring convention: normalise text (lowercase, strip
punctuation and extra whitespace), then compare. For questions with multiple
gold answers we take the max score over golds. Vietnamese uses whitespace
between syllables, so token = whitespace-delimited unit, same as SQuAD.
"""
import re
import string
from collections import Counter
from typing import List, Sequence

_PUNCT = set(string.punctuation)


def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    # Both empty -> perfect (both are "no answer"); one empty -> 0.
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(fn, prediction: str, golds: Sequence[str]) -> float:
    if not golds:
        # No gold answer (impossible question): correct iff prediction is empty.
        return float(fn(prediction, ""))
    return max(fn(prediction, g) for g in golds)


def evaluate(predictions: dict, references: dict) -> dict:
    """predictions: {id: str}, references: {id: [gold, ...]} -> {'EM', 'F1', 'count'}."""
    total_em = 0.0
    total_f1 = 0.0
    count = 0
    for qid, gold_list in references.items():
        pred = predictions.get(qid, "")
        total_em += metric_max_over_ground_truths(exact_match, pred, gold_list)
        total_f1 += metric_max_over_ground_truths(f1, pred, gold_list)
        count += 1
    if count == 0:
        return {"EM": 0.0, "F1": 0.0, "count": 0}
    return {"EM": 100.0 * total_em / count, "F1": 100.0 * total_f1 / count, "count": count}


def per_item_scores(predictions: dict, references: dict) -> dict:
    """{id: {'em': int, 'f1': float}} for downstream breakdown analysis."""
    out = {}
    for qid, gold_list in references.items():
        pred = predictions.get(qid, "")
        out[qid] = {
            "em": int(metric_max_over_ground_truths(exact_match, pred, gold_list)),
            "f1": metric_max_over_ground_truths(f1, pred, gold_list),
        }
    return out
