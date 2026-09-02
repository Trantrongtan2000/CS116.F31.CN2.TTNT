# -*- coding: utf-8 -*-
"""
Confusion Matrix Generator for CS116 Vietnamese MRC
Generates heatmap of prediction accuracy by error category
"""
import json
import re
import string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Text normalization
# ============================================================
def normalize_text(s):
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def compute_exact_match(pred, gt):
    return int(normalize_text(pred) == normalize_text(gt))

def compute_f1(pred, gt):
    pred_tokens = normalize_text(pred).split()
    gt_tokens = normalize_text(gt).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

# ============================================================
# TF-IDF Baseline Predictor
# ============================================================
class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
    
    def predict(self, context, question):
        sentences = [s.strip() for s in re.split(r"[.!?;\n]", context) if len(s.strip()) > 5]
        if not sentences:
            return context[:100]
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

# ============================================================
# Sample Classification (by difficulty)
# ============================================================
def classify_sample(context, question, answer):
    """Classify sample into difficulty category."""
    if len(normalize_text(answer).split()) <= 1:
        return "Ambiguous"
    if len(context.split()) > 250:
        return "Context Length"
    multi_hop_kw = ["tại sao", "vì sao", "do đó", "cho nên", "liệu có", "nguyên nhân", "hệ quả"]
    if any(kw in question.lower() for kw in multi_hop_kw):
        return "Multi-hop"
    return "Standard"

def classify_error(pred, gt):
    """Classify error type when prediction is wrong."""
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    pred_tokens = set(pred_norm.split())
    gt_tokens = set(gt_norm.split())
    
    if pred_tokens and gt_tokens:
        overlap = pred_tokens & gt_tokens
        if 0 < len(overlap) < min(len(pred_tokens), len(gt_tokens)):
            return "Word Boundary"
    return "Wrong"

# ============================================================
# Load Dataset
# ============================================================
def load_viquad(filepath, max_samples=500):
    """Load UIT-ViQuAD 2.0 format JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for article in data['data']:
        for para in article['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                qid = qa['id']
                question = qa['question']
                answer = None
                if qa.get('answers') and qa['answers'].get('text'):
                    answer = qa['answers']['text'][0]
                elif qa.get('plausible_answers') and qa['plausible_answers'].get('text'):
                    answer = qa['plausible_answers']['text'][0]
                if not answer:
                    continue
                samples.append({
                    'id': qid,
                    'context': context,
                    'question': question,
                    'answer': answer
                })
                if len(samples) >= max_samples:
                    return samples
    return samples

# ============================================================
# Generate Confusion Matrix
# ============================================================
def generate_confusion_matrix(test_file, output_png="confusion_matrix.png"):
    """Generate confusion matrix heatmap for MRC predictions."""
    print(f"[INFO] Loading test data from {test_file}")
    samples = load_viquad(test_file, max_samples=500)
    print(f"[INFO] Loaded {len(samples)} test samples")
    
    if not samples:
        print("[ERROR] No samples with ground truth answers found!")
        return [], np.zeros((4, 4), dtype=int)
    
    predictor = TFIDFPredictor()
    
    # Rows: True difficulty category
    # Columns: Prediction outcome
    row_categories = ["Standard", "Ambiguous", "Context Length", "Multi-hop"]
    col_categories = ["Correct", "Word Boundary", "Wrong"]
    
    confusion = np.zeros((len(row_categories), len(col_categories)), dtype=int)
    row_to_idx = {cat: i for i, cat in enumerate(row_categories)}
    col_to_idx = {cat: i for i, cat in enumerate(col_categories)}
    
    results = []
    for sample in samples:
        pred = predictor.predict(sample['context'], sample['question'])
        gt = sample['answer']
        
        # True category (difficulty)
        true_cat = classify_sample(sample['context'], sample['question'], gt)
        
        # Prediction outcome
        em = compute_exact_match(pred, gt)
        if em == 1:
            pred_cat = "Correct"
        else:
            pred_cat = classify_error(pred, gt)
        
        row_idx = row_to_idx.get(true_cat, 0)
        col_idx = col_to_idx.get(pred_cat, 2)
        confusion[row_idx][col_idx] += 1
        
        results.append({
            'question': sample['question'][:80],
            'true_cat': true_cat,
            'pred_cat': pred_cat,
            'em': em,
            'f1': round(compute_f1(pred, gt), 3)
        })
    
    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r['em'] == 1)
    avg_f1 = float(np.mean([r['f1'] for r in results]))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize by row for percentages
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    confusion_pct = confusion / row_sums * 100
    
    sns.heatmap(
        confusion_pct, 
        annot=True, 
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=col_categories,
        yticklabels=row_categories,
        ax=ax,
        cbar_kws={'label': 'Percentage (%)'}
    )
    
    ax.set_xlabel('Prediction Outcome', fontsize=12)
    ax.set_ylabel('Sample Difficulty Category', fontsize=12)
    ax.set_title(f'CS116 - Vietnamese MRC Confusion Matrix\n(TF-IDF Baseline | EM: {correct}/{total} = {100*correct/total:.1f}% | Avg F1: {avg_f1:.3f})', 
                 fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    print(f"[INFO] Confusion matrix saved to {output_png}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"CONFUSION MATRIX SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Correct (EM=1): {correct} ({100*correct/total:.1f}%)")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"\nCategory distribution:")
    for cat in row_categories:
        count = sum(1 for r in results if r['true_cat'] == cat)
        print(f"  {cat}: {count}")
    
    return results, confusion

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import os
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Validation set has ground truth answers; test set answers are null
    test_file = os.path.join(PROJECT_DIR, "viquad2_deduped_validation.json")
    output_file = os.path.join(PROJECT_DIR, "visualizations", "confusion_matrix.png")
    
    os.makedirs(os.path.join(PROJECT_DIR, "visualizations"), exist_ok=True)
    
    results, confusion = generate_confusion_matrix(test_file, output_file)
    
    # Also save numerical data
    data_file = os.path.join(PROJECT_DIR, "visualizations", "confusion_matrix_data.json")
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump({
            'matrix': confusion.tolist(),
            'row_categories': ["Standard", "Ambiguous", "Context Length", "Multi-hop"],
            'col_categories': ["Correct", "Word Boundary", "Wrong"],
            'total_samples': len(results),
            'correct': sum(1 for r in results if r['em'] == 1),
            'avg_f1': float(np.mean([r['f1'] for r in results]))
        }, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Matrix data saved to {data_file}")
