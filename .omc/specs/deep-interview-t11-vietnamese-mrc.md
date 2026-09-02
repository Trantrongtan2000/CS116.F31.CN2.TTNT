# Deep Interview Spec: T11 — Vietnamese MRC (CPU-only, inference-first rebuild)

## Metadata
- Interview ID: t11-vietnamese-mrc-rebuild
- Rounds: 5
- Final Ambiguity Score: 18%
- Type: brownfield (existing `CS116.F31.CN2.TTNT`, rebuilt clean)
- Generated: 2026-09-02
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.90 | 0.35 | 0.315 |
| Constraint Clarity | 0.80 | 0.25 | 0.200 |
| Success Criteria | 0.75 | 0.25 | 0.1875 |
| Context Clarity | 0.80 | 0.15 | 0.120 |
| **Total Clarity** | | | **0.8225** |
| **Ambiguity** | | | **0.1775** |

## Goal
Rebuild the CS116 T11 Vietnamese extractive MRC project as a clean, correct, CPU-runnable pipeline that produces **honest, reproducible results** — keeping only the UIT-ViQuAD 2.0 dataset from the old repo and discarding the rest. Given context + question, extract the answer span. Deliver: a TF-IDF baseline, a pretrained Vietnamese QA transformer used **inference-only** for the main results, a **small proof-of-concept CPU fine-tune** solely to generate a genuine training curve, a suite of non-training analyses, a working Streamlit demo, and a report.

## Constraints
- **CPU-only** — no reliable GPU. Full multi-epoch fine-tuning of transformers on the full 23k+ set is out of scope.
- **Internet available for a one-time model/download** from HuggingFace; the dataset is already local in the repo.
- Stay strictly **extractive MRC** — no generative QA, no RAG (original brief scope).
- Evaluate on the **validation split** — the deduplicated test split has null gold answers.
- Respect the transformer's max sequence length (e.g. XLM-R/PhoBERT ≤ 256–512 per model); no `max_length` exceeding the model's limit.
- Every reported metric must be backed by an artifact from an actual run — **no fabricated numbers** (the failure mode of the old repo).
- Deadline: unstated (does not affect the build plan).

## Non-Goals
- Training PhoBERT/XLM-R from scratch or full fine-tuning on GPU.
- ViDeBERTa or any model that is not actually run.
- BM25 (unless trivially added); the brief's "BM25" is optional next to TF-IDF.
- Generative answers, RAG, multi-document retrieval.
- Reproducing the old repo's inflated 68.5%/84.5% claims.

## Acceptance Criteria
- [ ] Dataset loads from local UIT-ViQuAD 2.0 with a leakage-free context-level split; an assertion fails the run on any cross-split context overlap.
- [ ] TF-IDF baseline runs and reports EM + token-F1 on the validation split.
- [ ] A best-fit pretrained Vietnamese extractive QA model (surveyed on HF; candidate `nguyenvulebinh/vi-mrc-base`) is loaded inference-only and reports **real** EM + token-F1 on the same validation samples.
- [ ] A small CPU fine-tune (tiny subset, few epochs) runs to completion and emits a genuine loss + EM/F1-per-epoch curve for the report.
- [ ] Non-training visualizations produced: EM/F1 bar comparison, EM/F1 vs context-length, question-type (single- vs multi-sentence) breakdown, outcome/confusion matrix.
- [ ] Streamlit demo launches without error, both baseline and transformer selectable via a shared `predict(context, question)` interface, returns and highlights an answer for arbitrary input.
- [ ] Report presents only measured numbers, states the eval split, and justifies the inference-only + tiny-curve methodology.
- [ ] Clean modular codebase (`dataset_loader`, baseline, model/inference, eval, app), `requirements.txt`, `README.md`; no leaked absolute paths, no duplicate large data files.

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "Train PhoBERT/XLM-R from scratch" (from brief) | CPU-only makes this infeasible | Pivot to pretrained inference-only for main results |
| Rebuild vs fix existing repo | Existing repo is broken + dishonest | Rebuild clean, keep only the dataset |
| Rubric needs per-epoch loss/EM/F1 curves | Inference-only has no training | Add a tiny CPU fine-tune purely for the curve + non-training analyses |
| Need a specific Vietnamese QA model | Which one, and is HF reachable? | Internet OK; model choice delegated to implementer (best size/speed fit) |
| Test split for final metrics | Test split has null answers | Evaluate on validation split, state it explicitly |

## Technical Context
Brownfield findings (from prior audit of `CS116.F31.CN2.TTNT`): the old pipeline never trained (crashed on a `TrainingArguments` kwarg), reported fabricated/contradictory metrics, and shipped a demo with a `SyntaxError` and a `predict()`/`predict_span()` mismatch. Salvageable and reused: the local `viquad2_*.json` dataset, the context-level dedup logic, and the EM/F1 metric implementation. Everything else is rebuilt. An OpenSpec change `remediate-cs116-mrc` exists but was scoped for *remediation with real GPU training* — it should be revised or replaced to match this inference-first, CPU-only rebuild.

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Context | core domain | text, length | has many Questions |
| Question | core domain | text, type (single/multi-sentence) | belongs to Context; has one AnswerSpan |
| AnswerSpan | core domain | text, start_idx, end_idx, is_impossible | extracted from Context |
| Dataset (UIT-ViQuAD 2.0) | external system | splits, SQuAD JSON | provides Context/Question/AnswerSpan |
| TF-IDF Baseline | supporting | vectorizer | predicts AnswerSpan (sentence-level) |
| Pretrained QA Model | external system | HF checkpoint, tokenizer | predicts AnswerSpan (inference-only) |
| Evaluation | supporting | EM, token-F1, by-length, by-type | scores predictions vs AnswerSpan |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 7 | 7 | - | - | N/A |
| 2 | 7 | 0 | 0 | 7 | 100% |
| 3 | 7 | 0 | 0 | 7 | 100% |
| 4 | 7 | 0 | 0 | 7 | 100% |
| 5 | 7 | 0 | 0 | 7 | 100% |

The domain model was stable from round 2 onward — all uncertainty was about *approach* (compute, inference vs training, rubric), not *entities*.

## Interview Transcript
<details>
<summary>Full Q&A (5 rounds)</summary>

### Round 1 — Goal
**Q:** Existing repo is broken — what is the goal of this effort?
**A:** Rebuild clean from scratch (keep only the dataset).
**Ambiguity:** 55%

### Round 2 — Constraints
**Q:** What compute + time budget for training?
**A:** Minimal / CPU-only.
**Ambiguity:** 48%

### Round 3 — Goal/approach
**Q:** On CPU, how should the transformer be realized?
**A:** Pretrained QA model, inference only.
**Ambiguity:** 37%

### Round 4 — Success Criteria (Contrarian)
**Q:** Rubric wants training curves but inference-only has none — how to satisfy visualization honestly?
**A:** Both — non-training analyses AND a tiny proof-of-concept training curve.
**Ambiguity:** 26%

### Round 5 — Constraints
**Q:** Internet for a one-time model download, and which model?
**A:** Yes internet; implementer picks the best-fit Vietnamese QA model.
**Ambiguity:** 18% (PASSED)

</details>
