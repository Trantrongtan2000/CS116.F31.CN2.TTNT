# -*- coding: utf-8 -*-
"""Generate all report figures from REAL run outputs (results/*.json).

Figures (written to visualizations/):
  1. em_f1_comparison.png     - EM/F1 bars, baseline vs transformer
  2. em_f1_by_length.png      - EM/F1 vs context-length bucket
  3. em_f1_by_question_type.png - single- vs multi-sentence reasoning
  4. outcome_matrix.png       - per-model outcome (Correct / Partial / Wrong) counts
  5. training_curve.png       - loss + EM/F1 per epoch (from finetune_tiny.py)

Reads only produced data; if an input JSON is missing it skips that figure and
says so. Nothing here fabricates numbers.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
VIZ_DIR = os.path.join(ROOT, "visualizations")


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"[SKIP] {name} not found — run the corresponding script first.")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_comparison(ev):
    models = list(ev["models"].keys())
    em = [ev["models"][m]["overall"]["EM"] for m in models]
    f1 = [ev["models"][m]["overall"]["F1"] for m in models]
    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    b1 = ax.bar(x - w / 2, em, w, label="Exact Match", color="#d62728")
    b2 = ax.bar(x + w / 2, f1, w, label="Token F1", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"EM / F1 on UIT-ViQuAD 2.0 validation (n={ev['num_questions']})")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "em_f1_comparison.png")


def fig_by_length(ev):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ("EM", "F1")):
        for m in ev["models"]:
            d = ev["models"][m]["by_context_length"]
            labels = list(d.keys())
            ax.plot(labels, [d[l][metric] for l in labels], marker="o", label=m)
        ax.set_title(f"{metric} vs context length"); ax.set_xlabel("Context length (words)")
        ax.set_ylabel(f"{metric} (%)"); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, "em_f1_by_length.png")


def fig_by_type(ev):
    models = list(ev["models"].keys())
    types = ["single-sentence", "multi-sentence"]
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(types)); w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        d = ev["models"][m]["by_question_type"]
        vals = [d.get(t, {}).get("F1", 0.0) for t in types]
        ax.bar(x + i * w, vals, w, label=m)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{t}\n(n={ev['models'][models[0]]['by_question_type'].get(t,{}).get('count',0)})"
                        for t in types])
    ax.set_ylabel("Token F1 (%)"); ax.set_ylim(0, 100)
    ax.set_title("F1 by question reasoning scope (heuristic)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "em_f1_by_question_type.png")


def fig_outcome_matrix(ev):
    """Correct (EM=1) / Partial (F1>0, EM=0) / Wrong (F1=0) per model, from samples+overall."""
    # Derive counts from per-item is not stored globally; reconstruct proportions
    # from overall EM/F1 is not exact, so we use the sample_predictions distribution
    # only as an illustration is avoided — instead we recompute from stored breakdown
    # counts which sum to n. We approximate Partial/Wrong via F1 vs EM gap per bucket.
    models = list(ev["models"].keys())
    cats = ["Correct", "Partial", "Wrong"]
    mat = np.zeros((len(models), 3))
    for i, m in enumerate(models):
        buckets = ev["models"][m]["by_context_length"].values()
        n = sum(b["count"] for b in buckets)
        correct = sum(b["EM"] / 100.0 * b["count"] for b in buckets)
        # tokens with any overlap ~ F1>0 proportion approximated by F1 mass >= small
        f1mass = sum(b["F1"] / 100.0 * b["count"] for b in buckets)
        partial = max(0.0, f1mass - correct)     # overlap beyond exact
        wrong = max(0.0, n - correct - partial)
        mat[i] = [correct, partial, wrong]
    fig, ax = plt.subplots(figsize=(8, 5))
    import seaborn as sns
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="YlOrRd",
                xticklabels=cats, yticklabels=models, ax=ax,
                cbar_kws={"label": "Questions"})
    ax.set_title("Prediction outcome distribution (approx. from EM/F1 mass)")
    _save(fig, "outcome_matrix.png")


def fig_training_curve(tc):
    curve = tc["curve"]
    ep = [c["epoch"] for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ep, [c["train_loss"] for c in curve], "b-o")
    axes[0].set_title(f"Training loss ({tc['model'].split('/')[-1]}, tiny CPU fine-tune)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, [c["val_em"] for c in curve], "r-o", label="Val EM")
    axes[1].plot(ep, [c["val_f1"] for c in curve], "g-s", label="Val F1")
    axes[1].set_title("Validation EM / F1 per epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)"); axes[1].set_ylim(0, 100); axes[1].grid(True, alpha=0.3); axes[1].legend()
    _save(fig, "training_curve.png")


def _save(fig, name):
    os.makedirs(VIZ_DIR, exist_ok=True)
    path = os.path.join(VIZ_DIR, name)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] wrote {os.path.relpath(path, ROOT)}")


def main():
    ev = _load("eval_results.json")
    if ev:
        fig_comparison(ev)
        fig_by_length(ev)
        fig_by_type(ev)
        fig_outcome_matrix(ev)
    tc = _load("training_curve.json")
    if tc:
        fig_training_curve(tc)
    print("Done.")


if __name__ == "__main__":
    main()
