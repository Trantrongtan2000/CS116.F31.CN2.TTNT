# -*- coding: utf-8 -*-
"""Build a curated set of QA test cases and run BOTH models on them.

Selects a diverse sample from the validation split — single-sentence factoid,
multi-sentence reasoning, long context, and unanswerable (impossible) — then runs
the TF-IDF baseline and the transformer, recording gold / prediction / EM / F1 for
each. Writes results/test_cases.json for the report to embed.

Every row is a real prediction from an actual run.
"""
import json
import os

from mrc import data, metrics
from mrc.baseline import TFIDFBaseline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_PER_CATEGORY = 3


def pick(examples, predicate, n):
    out = []
    for e in examples:
        if predicate(e):
            out.append(e)
        if len(out) >= n:
            break
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    val = data.load_split("validation")

    single = pick(val, lambda e: e.has_answer and e.question_type == "single-sentence"
                  and e.context_length < 180, N_PER_CATEGORY)
    multi = pick(val, lambda e: e.has_answer and e.question_type == "multi-sentence"
                 and e.context_length < 220, N_PER_CATEGORY)
    long_ctx = pick(val, lambda e: e.has_answer and e.context_length > 260, N_PER_CATEGORY)
    impossible = pick(val, lambda e: e.is_impossible, 2)

    groups = [("Single-sentence (factoid)", single),
              ("Multi-sentence (reasoning)", multi),
              ("Long context (>260 words)", long_ctx),
              ("Unanswerable (impossible)", impossible)]

    baseline = TFIDFBaseline()
    from mrc.qa_model import TransformerQA
    transformer = TransformerQA()

    cases = []
    for category, exs in groups:
        for e in exs:
            gold = e.answers[:1] if e.answers else []
            b_pred = baseline.predict(e.context, e.question)
            t_pred = transformer.predict(e.context, e.question)
            cases.append({
                "category": category,
                "id": e.id,
                "question": e.question,
                "gold": gold,
                "context_length": e.context_length,
                "context_preview": (e.context[:220] + "…") if len(e.context) > 220 else e.context,
                "baseline": {
                    "prediction": b_pred,
                    "em": int(metrics.metric_max_over_ground_truths(metrics.exact_match, b_pred, gold)),
                    "f1": round(metrics.metric_max_over_ground_truths(metrics.f1, b_pred, gold), 3),
                },
                "transformer": {
                    "prediction": t_pred,
                    "em": int(metrics.metric_max_over_ground_truths(metrics.exact_match, t_pred, gold)),
                    "f1": round(metrics.metric_max_over_ground_truths(metrics.f1, t_pred, gold), 3),
                },
            })
            print(f"[{category}] {e.question[:50]}...  base_f1="
                  f"{cases[-1]['baseline']['f1']} trans_f1={cases[-1]['transformer']['f1']}")

    out = os.path.join(RESULTS_DIR, "test_cases.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "model": transformer.model_name}, f,
                  ensure_ascii=False, indent=2)
    print(f"[INFO] wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
