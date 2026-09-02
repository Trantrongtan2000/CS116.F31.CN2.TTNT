"""CS116 T11 - Vietnamese Extractive MRC (clean, CPU-only, inference-first rebuild).

Public modules:
- data:     load UIT-ViQuAD 2.0, leakage-free split, question-type / length tagging
- metrics:  SQuAD-style Exact Match and token-level F1
- baseline: TF-IDF sentence-retrieval baseline
- qa_model: pretrained Vietnamese QA model, inference-only, shared predict() interface
"""
__all__ = ["data", "metrics", "baseline", "qa_model"]
