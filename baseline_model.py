# -*- coding: utf-8 -*-
"""
Baseline Model: TF-IDF Context Sentence Matching for Extractive QA
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BaselineTFIDFQA:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def predict(self, context, question):
        sentences = [s.strip() for s in re.split(r"[.?!\n]", context) if len(s.strip()) > 5]
        if not sentences:
            return context[:50]

        corpus = sentences + [question]
        try:
            tfidf_mat = self.vectorizer.fit_transform(corpus)
            sent_vecs = tfidf_mat[:-1]
            q_vec = tfidf_mat[-1]
            sims = cosine_similarity(sent_vecs, q_vec).flatten()
            best_idx = int(sims.argmax())
            return sentences[best_idx]
        except Exception:
            return sentences[0]
