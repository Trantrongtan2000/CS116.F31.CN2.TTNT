# -*- coding: utf-8 -*-
"""Pretrained Vietnamese extractive QA model — INFERENCE ONLY.

We do not fine-tune here (CPU-only constraint). We load a model already
fine-tuned for Vietnamese MRC from the HuggingFace Hub and run it through the
transformers `question-answering` pipeline, which handles the details that the
old repo got wrong:
  - offset mapping (char-accurate span extraction)
  - doc-stride windowing for contexts longer than the model's max length
  - optional no-answer handling for impossible questions

Default model: `deepset/xlm-roberta-base-squad2` — a multilingual XLM-R model
fine-tuned for extractive QA on SQuAD 2.0. It is NOT gated (loads without auth),
CPU-runnable, and covers Vietnamese via XLM-R's multilingual pretraining, so this
is an honest zero-shot cross-lingual application to UIT-ViQuAD. The stronger
Vietnamese-specific checkpoints (e.g. nguyenvulebinh/vi-mrc-base) are gated behind
HuggingFace auth; set MRC_QA_MODEL to use one if you have access.

Exposes `predict(context, question)` — the shared interface (same signature as the
TF-IDF baseline) so the demo and evaluator are model-agnostic.
"""
import os
from typing import Optional

DEFAULT_MODEL = os.environ.get("MRC_QA_MODEL", "deepset/xlm-roberta-base-squad2")


class TransformerQA:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 max_seq_len: int = 384, doc_stride: int = 128,
                 handle_impossible: bool = True, null_threshold: float = 0.0):
        self.model_name = model_name
        self.name = f"Transformer ({model_name.split('/')[-1]})"
        self.handle_impossible = handle_impossible
        self.null_threshold = null_threshold
        self._pipe = None
        self._max_seq_len = max_seq_len
        self._doc_stride = doc_stride

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForQuestionAnswering, pipeline,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        # Respect the model's real limit; never exceed model_max_length.
        model_max = getattr(tokenizer, "model_max_length", 512)
        if not isinstance(model_max, int) or model_max > 100000:
            model_max = 512
        self._max_seq_len = min(self._max_seq_len, model_max)
        self._pipe = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
            framework="pt",
        )

    def predict(self, context: str, question: str) -> str:
        self._ensure_loaded()
        if not context.strip() or not question.strip():
            return ""
        try:
            result = self._pipe(
                question=question,
                context=context,
                max_seq_len=self._max_seq_len,
                doc_stride=self._doc_stride,
                handle_impossible_answer=self.handle_impossible,
                max_answer_len=64,
            )
        except Exception as exc:  # keep evaluation robust on odd inputs
            print(f"[WARN] QA inference failed: {exc}")
            return ""
        answer = (result.get("answer") or "").strip()
        # Below-threshold score => treat as no answer (for impossible questions).
        if self.handle_impossible and result.get("score", 1.0) < self.null_threshold:
            return ""
        return answer
