# CS116 — Vietnamese Extractive MRC (T11)

Đề tài T11: **Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt** (Vietnamese
Extractive Machine Reading Comprehension). Cho một đoạn **context** và một
**question**, hệ thống trích xuất **answer span** nằm trong context.

This is a clean, CPU-runnable, **inference-first** rebuild. Every number reported
here and in `results/` is produced by an actual run over the real dataset — there
are no hard-coded or placeholder metrics anywhere in this project.

## Approach

| Component | What it is |
|---|---|
| **Baseline** | TF-IDF + cosine sentence retrieval (`mrc/baseline.py`) |
| **Transformer** | `deepset/xlm-roberta-base-squad2` (multilingual XLM-R, SQuAD2), loaded **inference-only** via the HuggingFace QA pipeline — proper offset mapping + doc-stride windowing (`mrc/qa_model.py`) |
| **Tiny fine-tune** | A small CPU fine-tune of a compact multilingual model, used **only** to produce a genuine loss / EM-F1-per-epoch curve for the report (`finetune_tiny.py`) |

Why inference-only: the target environment is **CPU-only**, where full fine-tuning
of a transformer on 28k+ questions is infeasible. We therefore run a model already
fine-tuned for extractive QA and report its real EM/F1 on Vietnamese. The stronger
Vietnamese-specific checkpoints (e.g. `nguyenvulebinh/vi-mrc-base`) are gated behind
HuggingFace auth; set `MRC_QA_MODEL` to use one if you have access.

## Dataset

**UIT-ViQuAD 2.0** (SQuAD-2.0 JSON), loaded locally from `viquad2_deduped_*.json`.
Splits are **context-level deduplicated** (no context appears in two splits — a
`assert_no_leakage` check fails the run otherwise). Evaluation runs on the
**validation** split, because the deduplicated **test** split ships with null gold
answers (no EM/F1 is computable there) — this is stated wherever numbers appear.

## Setup

```bash
# Python 3.12 recommended (torch has no 3.14 wheels yet)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
# 1) Evaluate baseline + transformer on the validation split (CPU-friendly subset)
python run_eval.py --limit 400
#    ...or the whole split (slower):  python run_eval.py --full
#    ...baseline only:                python run_eval.py --models baseline

# 2) Tiny fine-tune -> genuine training curve for the report
python finetune_tiny.py --train-size 200 --val-size 100 --epochs 3

# 3) Build all figures from the produced results/
python make_visualizations.py

# 4) Interactive demo
streamlit run app_streamlit.py
```

## Outputs

- `results/eval_results.json` — EM/F1 per model, plus breakdowns by context-length
  bucket and by question reasoning-scope (single- vs multi-sentence), with sample
  predictions.
- `results/training_curve.json` — loss + EM/F1 per epoch from the tiny fine-tune.
- `visualizations/*.png` — EM/F1 comparison, EM/F1 vs context length, F1 by
  question type, outcome matrix, and the training curve.

## Project structure

```
mrc/
  data.py        # load UIT-ViQuAD 2.0, leakage-free split, question-type/length tags
  metrics.py     # Exact Match + token-level F1 (SQuAD convention)
  baseline.py    # TF-IDF sentence-retrieval baseline
  qa_model.py    # pretrained QA model, inference-only, shared predict() interface
run_eval.py           # evaluate models -> results/eval_results.json
finetune_tiny.py      # tiny CPU fine-tune -> results/training_curve.json
make_visualizations.py# figures from results/
app_streamlit.py      # web demo
```

## Metrics

- **Exact Match (EM):** prediction matches a gold answer exactly after
  normalisation (lowercase, strip punctuation/whitespace).
- **Token-level F1:** harmonic mean of token precision/recall vs the gold answer;
  max over multiple golds.

## Team — Nhóm 7

- Lê Quang Thi (25210337)
- Trần Trọng Tấn (25210334)
- Nguyễn Quang Lâm (25210289)
- Võ Cẩm Thu (25210342)

GVHD: ThS. Nguyễn Hữu Quyền — Trường ĐH Công nghệ Thông tin, ĐHQG-HCM.
