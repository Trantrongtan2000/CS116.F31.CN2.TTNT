# -*- coding: utf-8 -*-
"""UIT-ViQuAD 2.0 loading, leakage-free splitting, and per-question tagging.

The dataset ships in SQuAD-2.0-style JSON:
    data[].paragraphs[].qas[] with fields
        id, question, is_impossible,
        answers        = {"text": [...], "answer_start": [...]}
        plausible_answers (only for impossible questions)

We flatten to `Example` records and tag each with:
- context_length   (word count) -> for the EM/F1-vs-length analysis
- question_type    ("single-sentence" | "multi-sentence") -> reasoning-scope analysis

The question-type tag is a documented HEURISTIC proxy (no gold reasoning-scope
label exists in ViQuAD): we locate the sentence containing the gold answer and
measure how many of the question's content words already appear in that single
sentence. High overlap => the answer is supported by one sentence
("single-sentence"); low overlap => evidence is spread across sentences
("multi-sentence"). Impossible questions are excluded from this tag.
"""
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Vietnamese + generic high-frequency function words to ignore when measuring
# question/answer-sentence content overlap.
_STOPWORDS = {
    "là", "và", "của", "có", "được", "cho", "trong", "một", "các", "những",
    "đã", "khi", "nào", "gì", "ai", "bao", "nhiêu", "sao", "thế", "ở", "với",
    "để", "này", "đó", "ra", "vào", "về", "từ", "theo", "hay", "hoặc", "thì",
    "mà", "như", "nên", "bị", "sẽ", "còn", "cũng", "đến", "tại", "trên", "dưới",
    "the", "a", "an", "of", "in", "is", "what", "who", "when", "where", "how",
}

# Cue words that typically require reasoning across sentences.
_MULTIHOP_CUES = (
    "tại sao", "vì sao", "nguyên nhân", "hệ quả", "như thế nào",
    "bằng cách nào", "so với", "khác nhau", "do đâu", "dẫn đến",
)


@dataclass
class Example:
    id: str
    title: str
    context: str
    question: str
    answers: List[str]          # gold answer texts ([] if impossible)
    answer_starts: List[int]
    is_impossible: bool
    plausible_answers: List[str] = field(default_factory=list)
    context_length: int = 0     # word count
    question_type: Optional[str] = None  # single-sentence | multi-sentence | None(impossible)

    @property
    def has_answer(self) -> bool:
        return not self.is_impossible and len(self.answers) > 0


def load_squad_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _content_words(text: str) -> set:
    words = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
                   " ", text.lower()).split()
    return {w for w in words if w and w not in _STOPWORDS and len(w) > 1}


def classify_question_type(context: str, question: str,
                           answer: str, answer_start: int) -> str:
    """Heuristic single- vs multi-sentence reasoning scope. See module docstring."""
    q_lower = question.lower()
    if any(cue in q_lower for cue in _MULTIHOP_CUES):
        return "multi-sentence"

    sentences = _split_sentences(context)
    if not sentences:
        return "single-sentence"

    # Find the sentence that contains the answer (by character offset if valid,
    # else by substring match).
    answer_sentence = sentences[0]
    if 0 <= answer_start < len(context):
        running = 0
        for s in sentences:
            idx = context.find(s, running)
            if idx == -1:
                idx = running
            if idx <= answer_start < idx + len(s):
                answer_sentence = s
                break
            running = idx + len(s)
    else:
        for s in sentences:
            if answer and answer in s:
                answer_sentence = s
                break

    q_words = _content_words(question)
    if not q_words:
        return "single-sentence"
    sent_words = _content_words(answer_sentence)
    overlap = len(q_words & sent_words) / len(q_words)
    # >=50% of question content is in the answer's own sentence -> single-sentence.
    return "single-sentence" if overlap >= 0.5 else "multi-sentence"


def to_examples(squad: dict) -> List[Example]:
    examples: List[Example] = []
    for article in squad.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para["context"]
            ctx_len = len(context.split())
            for qa in para.get("qas", []):
                ans = qa.get("answers") or {}
                texts = list(ans.get("text", []) or [])
                starts = list(ans.get("answer_start", []) or [])
                is_imp = bool(qa.get("is_impossible", False)) or len(texts) == 0
                plausible = []
                pa = qa.get("plausible_answers")
                if isinstance(pa, dict):
                    plausible = list(pa.get("text", []) or [])

                qtype = None
                if not is_imp and texts:
                    qtype = classify_question_type(
                        context, qa["question"], texts[0],
                        starts[0] if starts else -1,
                    )

                examples.append(Example(
                    id=qa["id"],
                    title=title,
                    context=context,
                    question=qa["question"],
                    answers=texts,
                    answer_starts=starts,
                    is_impossible=is_imp,
                    plausible_answers=plausible,
                    context_length=ctx_len,
                    question_type=qtype,
                ))
    return examples


def load_split(split: str, data_dir: str = DATA_DIR, deduped: bool = True) -> List[Example]:
    """Load 'train' | 'validation' | 'test' from local SQuAD-format JSON."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"split must be train/validation/test, got {split!r}")
    prefix = "viquad2_deduped_" if deduped else "viquad2_"
    path = os.path.join(data_dir, f"{prefix}{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return to_examples(load_squad_json(path))


def contexts_of(examples: List[Example]) -> set:
    return {e.context for e in examples}


def assert_no_leakage(*splits: List[Example]) -> None:
    """Fail loudly if any context appears in more than one split."""
    ctx_sets = [contexts_of(s) for s in splits]
    for i in range(len(ctx_sets)):
        for j in range(i + 1, len(ctx_sets)):
            overlap = ctx_sets[i] & ctx_sets[j]
            assert not overlap, (
                f"Data leakage: {len(overlap)} context(s) shared between "
                f"split #{i} and split #{j}"
            )


def references_dict(examples: List[Example]) -> Dict[str, List[str]]:
    """{id: [gold, ...]} for metric scoring ([] when impossible)."""
    return {e.id: e.answers for e in examples}


def subset(examples: List[Example], n: Optional[int], seed: int = 42) -> List[Example]:
    if n is None or n >= len(examples):
        return examples
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    return [examples[i] for i in sorted(idx[:n])]
