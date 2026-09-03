# CS116 Vietnamese MRC — Documentation & Gap Audit

**Repository:** `CS116.F31.CN2.TTNT`
**Subject:** CS116 — Lập trình Python cho Máy học (Đề tài T11: Vietnamese Extractive MRC)
**Audit date:** 2026-09-02
**Scope:** Full source tree at commit `af9de6f` — code, data artifacts, report, and reproducibility of reported results.

---

## 1. Executive summary

The project **scaffolding is competent** — a dataset loader with genuine context-level deduplication, clean EM/F1 metrics, sensible ROCm plumbing, and a real TF-IDF confusion matrix. But the **headline deliverables do not exist**:

- The PhoBERT model was **never successfully trained** (the only training run on record crashed before training started).
- The reported accuracy figures (**68.5% EM / 84.5% F1**) are **fabricated** — no run produced them, and they contradict each other across files and contradict the one real measured artifact.
- The Streamlit demo — the instructor's stated #1 grading criterion — **does not run** (hard `SyntaxError`).
- **ViDeBERTa** appears in the results table but is **not implemented** anywhere.

In short: the project *documents results it never produced*. The remediation is achievable — the pipeline is close — but the current submission's central claims are unsupported.

---

## 2. Architecture as built

```
                    +---------------------------+
   UIT-ViQuAD 2.0   |   dataset_loader.py       |
   (local JSON /    |  - load_viquad2_0()       |
    HF fallback)    |  - deduplicate_contexts() |  context-level split
                    |  - split_dataset_by_ctx() |  (anti-leakage)
                    +------------+--------------+
                                 |
                 +---------------+----------------+
                 v                                v
      +---------------------+          +-------------------------+
      | baseline_model.py   |          | train_phobert_full.py   |
      | TF-IDF + cosine     |          | PhoBERT QA fine-tune    |  <-- CRASHED, never ran
      | (sentence retrieval)|          | + evaluate + plots      |
      +----------+----------+          +-----------+-------------+
                 |                                 |
                 |          +----------------------+
                 v          v
      +----------------------------+     +--------------------------+
      | eval_metrics.py            |     | train_phobert_qa.py      |
      | EM / token-F1              |     | VietnameseQAModel        |  <-- train() is a STUB
      | error classification        |     | (inference wrapper)      |
      +----------------------------+     +-----------+--------------+
                                                     |
                    +--------------------------------+
                    v                                v
       +--------------------------+     +---------------------------+
       | confusion_matrix.py      |     | app_streamlit.py          |  <-- SYNTAX ERROR
       | (TF-IDF only, real run)  |     | web demo                  |
       +--------------------------+     +---------------------------+
```

### File inventory

| File | Role | State |
| :--- | :--- | :--- |
| `dataset_loader.py` | Load UIT-ViQuAD 2.0, context-level split, dedup | Works; good design |
| `baseline_model.py` | TF-IDF + cosine sentence retrieval | Works |
| `eval_metrics.py` | EM / token-F1, error classification | Works; `detailed_error_analysis` never called |
| `train_phobert_full.py` | Real PhoBERT fine-tune + eval + plots | **Crashed before training** |
| `train_phobert_qa.py` | Inference wrapper + `train_phobert_qa()` | **Training fn is a stub** |
| `confusion_matrix.py` | TF-IDF confusion matrix over 500 val samples | Works; only real result artifact |
| `app_streamlit.py` | Web demo | **SyntaxError — will not parse** |
| `error_analysis.md` | Error taxonomy | Hand-authored, not generated |
| `Bao_Cao_Do_An_CS116.*` | Written report (md/docx/pdf/html) | Contains fabricated metrics |
| `train_colab.ipynb`, `test_rocm_amd.py` | Colab notebook, ROCm tests | Unverified |

---

## 3. Central finding — the headline results are unsubstantiated

The README and report present these as experimental results:

| Source | TF-IDF EM/F1 | PhoBERT EM/F1 | ViDeBERTa EM/F1 |
| :--- | :---: | :---: | :---: |
| `README.md` | 28.7 / 39.4 | **68.5 / 84.5** | 71.2 / 86.1 |
| `Bao_Cao_Do_An_CS116.md` | 28.4 / 46.2 | **68.7 / 84.5** | — |
| `app_streamlit.py` | ~25 / ~35 | ~62 / ~74 | ~65 / ~77 |

Every number disagrees with the others, and **none of them came from a run.** Evidence:

1. **The only real trainer crashed.** `training_output.log` shows `train_phobert_full.py` dying at `TrainingArguments(... pin_memory=...)` → `TypeError: unexpected keyword argument 'pin_memory'`, *before training started*. (The crash log predates a patch to `dataloader_pin_memory`; no later log shows a completed run.)
2. **No output artifacts exist.** There is no `models/` directory, no `training_history.json`, no `test_predictions.json`, no `training_report.json`, and no training-curve PNG — all of which `main()` would have written on success.
3. **The second trainer is a stub.** `train_phobert_qa.py::train_phobert_qa()` builds the model and `TrainingArguments`, then `return model, tokenizer, training_history` — it **never calls `Trainer.train()`**.
4. **The one real result contradicts the claim.** `visualizations/confusion_matrix_data.json` records the actual TF-IDF run: **3/500 = 0.6% EM, avg F1 0.231** on validation — versus the "28.7% EM" claimed for TF-IDF in the same README (a ~48x discrepancy).
5. **ViDeBERTa is vaporware.** It appears in the results table and future-work section but is **not implemented** in any script.

**Conclusion:** PhoBERT was never trained, never evaluated; the "68.5% EM / 84.5% F1" figures are invented.

---

## 4. Gap register

### Critical — blocks the core deliverables

| # | Gap | Evidence |
| :---: | :--- | :--- |
| C1 | PhoBERT never trained/evaluated; no weights, no metrics, crash-only log | `training_output.log`, absent `models/` |
| C2 | Streamlit demo won't parse — `SyntaxError` at `app_streamlit.py:21` (unterminated string: `"PhoBERT Transformer (Full)', '...'`) | `python3 -m ast` fails |
| C3 | PhoBERT demo path doubly broken — app calls `qa_engine.predict()`, but `VietnameseQAModel` only defines `predict_span()` → `AttributeError` even after C2 is fixed | `train_phobert_qa.py:51` |
| C4 | Reported metrics mutually contradictory and contradicted by the one real artifact (0.6% vs 28.7% EM for TF-IDF) | Section 3 |

### High — correctness / methodology

| # | Gap | Evidence |
| :---: | :--- | :--- |
| H1 | `max_length=384` exceeds PhoBERT's 256-token limit → inference error / silent misbehavior | `train_phobert_qa.py:57`, `train_phobert_full.py:53` |
| H2 | `doc_stride`/overflow half-implemented: training uses `return_overflowing_tokens`, but eval truncates to one window (no stride) → answers past the window unreachable; train/eval preprocessing diverge | `preprocess_function` vs `evaluate_model` |
| H3 | Fragile span alignment: answer positions found by exact token-subsequence match; on failure defaults `start=end=0` (points at `<s>`), silently training bad labels | `train_phobert_full.py:219-237` |
| H4 | Unanswerable questions (~32% of train per `split_stats`) all mapped to position 0 with no answerability head/threshold | `preprocess_function`; `error_analysis.md` E5 |
| H5 | Split statistics inconsistent across files (train 4101 vs 138 contexts; README quotes raw 3,814/7,301 while claiming dedup split) | `viquad2_final_stats.json` vs `viquad2_split_stats.json` vs README |
| H6 | Test set has no answers (all null, `impossible_pct: 0.0`) → "test EM/F1" cannot exist; eval only possible on validation | `evaluate_model` "No ground truth" branch |

### Medium — reproducibility / hygiene

| # | Gap | Evidence |
| :---: | :--- | :--- |
| M1 | `error_analysis.md` hand-authored; its categories don't match `classify_error()` output; `detailed_error_analysis()` never invoked | `eval_metrics.py` |
| M2 | `torch_compile=True` for ROCm likely fails on RX 6700 XT (gfx1030) toolchain; unguarded | `train_phobert_full.py:575` |
| M3 | Two overlapping trainers with different configs (batch 16 vs 8+accum); README tells users to run the stub | README Quick Start |
| M4 | Duplicate ~19 MB data files committed (`viquad2_train.json` == `viquad2_deduped_train.json`); large JSON in git; no `.gitignore` | repo listing |
| M5 | Team roster inconsistent across README, report, `context.md` | multiple files |

### Low

| # | Gap | Evidence |
| :---: | :--- | :--- |
| L1 | Typos in demo sample ("Thủ tướng Chính **phát**") | `app_streamlit.py:43` |
| L2 | Hardcoded absolute path `/home/tan/...` leaked in `context.md` and log | `context.md`, `training_output.log` |
| L3 | `train_colab.ipynb` and `test_rocm_amd.py` unverified | — |

---

## 5. Recommended remediation roadmap

Ordered to restore integrity fastest.

1. **Make one claim true before any other work.** Train PhoBERT for real (Colab GPU, `USE_FULL_DATASET=1`), capturing `training_history.json`, `test_predictions.json`, and curves. Everything else is downstream of this.
2. **Fix the trainer before running it:** set `max_length<=256` (H1), guard/remove `torch_compile` (M2), harden span alignment + count alignment failures (H3), and decide on answerability handling (H4).
3. **Evaluate on validation, not "test"** (H6), and **replace every reported number** with the measured ones (C4). If ViDeBERTa is not implemented, remove it from results (Section 3).
4. **Repair the demo:** fix the `SyntaxError` (C2) and the `predict()`/`predict_span()` mismatch (C3); confirm it launches.
5. **Regenerate `error_analysis.md`** from `detailed_error_analysis()` on real predictions (M1).
6. **Reconcile dataset statistics** to a single source of truth (H5), delete duplicate data + add `.gitignore` (M4), unify the two trainers (M3), fix roster/typos/paths (M5, L1, L2).

---

## 6. Honest bottom line

The engineering *around* the model is largely sound and the gap to a genuine result is not large. But as submitted, the project's three core claims — a trained PhoBERT model, its 68.5%/84.5% accuracy, and a working demo — are each unsupported or broken. Grading criteria that reward a live demo and visualized real metrics are currently unmet. Fixing C1–C4 and H1–H2 would move this from "documents fictional results" to "reproducible."
