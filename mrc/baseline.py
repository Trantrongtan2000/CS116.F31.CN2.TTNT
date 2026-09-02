# -*- coding: utf-8 -*-
"""TF-IDF sentence-retrieval baseline for extractive QA.

Given a context and a question, split the context into sentences, TF-IDF-vectorise
sentences + question, and return the sentence with the highest cosine similarity to
the question. This is a lower-bound benchmark: it retrieves a whole sentence rather
than a minimal span, so Exact Match is expected to be low while F1 captures partial
token overlap.

Exposes `predict(context, question)` — the shared interface used by the demo and
the evaluator, identical in signature to the transformer model.
"""
import re
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFBaseline:
    name = "TF-IDF Baseline"

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features

    @staticmethod
    def _sentences(context: str) -> List[str]:
        parts = re.split(r"(?<=[.!?…])\s+|\n+|;", context)
        return [s.strip() for s in parts if len(s.strip()) > 5]

    def predict(self, context: str, question: str) -> str:
        sentences = self._sentences(context)
        if not sentences:
            return context[:100].strip()
        # Fit per-example: the corpus is tiny and this keeps the baseline stateless.
        vectorizer = TfidfVectorizer(max_features=self.max_features)
        try:
            matrix = vectorizer.fit_transform(sentences + [question])
        except ValueError:
            return sentences[0]
        sims = cosine_similarity(matrix[:-1], matrix[-1]).flatten()
        return sentences[int(sims.argmax())]
