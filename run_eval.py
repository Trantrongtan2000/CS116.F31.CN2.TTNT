# -*- coding: utf-8 -*-
"""Evaluate models on the UIT-ViQuAD 2.0 VALIDATION split and write real metrics.

Why validation and not test: the deduplicated test split ships with null gold
answers (is_impossible / empty), so EM/F1 cannot be computed there. All headline
numbers therefore come from the validation split, and this is stated in the report.

Outputs (under results/):
  - eval_results.json : overall EM/F1 per model + breakdowns by context-length
                        bucket and by question-type, plus per-item outcomes.
Everything written here is produced by an actual run over real data — no
hard-coded numbers anywhere in this project.

Usage:
  python run_eval.py --limit 400            # subset for speed on CPU
  python run_eval.py --full                 # whole validation split
  python run_eval.py --models baseline      # skip the transformer
"""
import argparse
import json
import os
import time

from mrc import data, metrics
from mrc.baseline import TFIDFBaseline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

LENGTH_BUCKETS = [(0, 100), (100, 200), (200, 300), (300, 10 ** 9)]
LENGTH_LABELS = ["<100", "100-200", "200-300", "300+"]


def bucket_label(n: int) -> str:
    for (lo, hi), lab in zip(LENGTH_BUCKETS, LENGTH_LABELS):
        if lo <= n < hi:
            return lab
    return LENGTH_LABELS[-1]


def run_model(model, examples) -> dict:
    preds = {}
    t0 = time.time()
    for i, ex in enumerate(examples):
        preds[ex.id] = model.predict(ex.context, ex.question)
        if (i + 1) % 50 == 0:
            print(f"    {model.name}: {i + 1}/{len(examples)} "
                  f"({(time.time() - t0):.0f}s)")
    elapsed = time.time() - t0
    return preds, elapsed


def breakdowns(examples, per_item) -> dict:
    """EM/F1 grouped by context-length bucket and by question-type."""
    by_len, by_type = {}, {}
    for ex in examples:
        blab = bucket_label(ex.context_length)
        by_len.setdefault(blab, []).append(per_item[ex.id])
        if ex.question_type:  # answerable only
            by_type.setdefault(ex.question_type, []).append(per_item[ex.id])

    def agg(items):
        n = len(items)
        if n == 0:
            return {"EM": 0.0, "F1": 0.0, "count": 0}
        return {
            "EM": 100.0 * sum(x["em"] for x in items) / n,
            "F1": 100.0 * sum(x["f1"] for x in items) / n,
            "count": n,
        }

    return {
        "by_context_length": {k: agg(by_len.get(k, [])) for k in LENGTH_LABELS},
        "by_question_type": {k: agg(v) for k, v in by_type.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400,
                    help="Number of validation examples (CPU-friendly). Ignored with --full.")
    ap.add_argument("--full", action="store_true", help="Use the entire validation split.")
    ap.add_argument("--models", nargs="+", default=["baseline", "transformer"],
                    choices=["baseline", "transformer"])
    ap.add_argument("--model-name", default=None, help="Override the HF QA checkpoint.")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/4] Loading validation split...")
    val = data.load_split("validation")
    # Leakage guard against train/test.
    train = data.load_split("train")
    test = data.load_split("test")
    data.assert_no_leakage(train, val, test)
    print(f"      No-leakage check passed. Validation has {len(val)} questions "
          f"({sum(e.is_impossible for e in val)} impossible).")

    examples = val if args.full else data.subset(val, args.limit)
    references = data.references_dict(examples)
    print(f"[2/4] Evaluating on {len(examples)} questions "
          f"({'full' if args.full else 'subset'}).")

    report = {
        "dataset": "UIT-ViQuAD 2.0 (deduplicated)",
        "split": "validation",
        "num_questions": len(examples),
        "num_impossible": sum(e.is_impossible for e in examples),
        "models": {},
        "provenance": "All metrics computed live by run_eval.py — no hard-coded values.",
    }

    models = []
    if "baseline" in args.models:
        models.append(TFIDFBaseline())
    if "transformer" in args.models:
        from mrc.qa_model import TransformerQA, DEFAULT_MODEL
        models.append(TransformerQA(args.model_name or DEFAULT_MODEL))

    for model in models:
        print(f"[3/4] Running {model.name} ...")
        preds, elapsed = run_model(model, examples)
        overall = metrics.evaluate(preds, references)
        per_item = metrics.per_item_scores(preds, references)
        report["models"][model.name] = {
            "overall": overall,
            "avg_latency_ms": round(1000.0 * elapsed / max(1, len(examples)), 1),
            **breakdowns(examples, per_item),
            "sample_predictions": [
                {"question": e.question, "gold": e.answers[:1],
                 "prediction": preds[e.id], "context_length": e.context_length,
                 "question_type": e.question_type}
                for e in examples[:15]
            ],
        }
        print(f"      {model.name}: EM={overall['EM']:.2f}  F1={overall['F1']:.2f}  "
              f"(n={overall['count']}, {report['models'][model.name]['avg_latency_ms']}ms/q)")

    out = os.path.join(RESULTS_DIR, "eval_results.json")
    print(f"[4/4] Writing {out}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
