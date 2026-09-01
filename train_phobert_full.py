# -*- coding: utf-8 -*-
"""
Full Training Script for PhoBERT QA on UIT-ViQuAD 2.0
Course: CS116 - Do an T11 (He thong doc hieu va tra loi cau hoi tieng Viet)
Author: Nhom 7 (Le Quang Thi, Tran Trong Tan, Nguyen Quang Lam, Vo Cam Thu)

Training Configuration:
  - Model: vinai/phobert-base-v2
  - LR: 3e-5, Batch: 16, Epochs: 3
  - FP16 enabled for GPU (auto-disabled on CPU)
"""
import os
import json
import time
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from eval_metrics import compute_exact_match, compute_f1

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "vinai/phobert-base-v2"
TRAIN_FILE = os.path.join(PROJECT_DIR, "viquad2_deduped_train.json")
VAL_FILE = os.path.join(PROJECT_DIR, "viquad2_deduped_validation.json")
TEST_FILE = os.path.join(PROJECT_DIR, "viquad2_deduped_test.json")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "models/phobert-viquad2-full")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
VIZ_DIR = os.path.join(PROJECT_DIR, "visualizations")

# Training hyperparameters (as specified in task)
LEARNING_RATE = 3e-5
# Batch size optimized for 12GB VRAM (AMD Radeon RX 6700 XT)
# PhoBERT-base ~135M params, FP16: ~270MB model + ~1.5GB activations per sample at max_length=384
# 12GB VRAM: batch_size=8 with gradient_accumulation=2 → effective batch=16
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8 * 2 = 16
NUM_EPOCHS = 3
MAX_LENGTH = 384
DOC_STRIDE = 128
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# CPU mode: use subset for feasible training time
USE_FULL_DATASET = os.environ.get("USE_FULL_DATASET", "0") == "1"
TRAIN_SUBSET_SIZE = 5000
VAL_SUBSET_SIZE = 500

# ============================================================
# DEVICE DETECTION: CUDA / ROCm / CPU
# ============================================================
def detect_device():
    """Detect available device: CUDA (NVIDIA), ROCm (AMD HIP), or CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "GPU"
        # Check if it's an AMD GPU under ROCm (HIP backend)
        is_rocm = any(keyword in device_name.lower() for keyword in ["amd", "radeon", "gfx", "rx "])
        is_cuda = not is_rocm  # NVIDIA CUDA
        device = torch.device("cuda:0")
        return device, True, is_rocm, device_name
    else:
        return torch.device("cpu"), False, False, "CPU"

DEVICE, IS_GPU, IS_ROCM, GPU_NAME = detect_device()

# Configure environment for AMD ROCm
if IS_ROCM:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    # RX 6700 XT = gfx1030, override for HIP compatibility
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512")
    # Enable HIP graph and memory pooling
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    # TF32 for AMD RDNA2/RDNA3 matmul acceleration
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] ROCm (AMD) mode enabled on {GPU_NAME}")
    print(f"[INFO] HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
elif IS_GPU:
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] CUDA (NVIDIA) mode enabled on {GPU_NAME}")

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] GPU available: {IS_GPU}")
print(f"[INFO] GPU name: {GPU_NAME}")
print(f"[INFO] ROCm (AMD): {IS_ROCM}")
print(f"[INFO] Use full dataset: {USE_FULL_DATASET}")
if not IS_GPU:
    print(f"[INFO] CPU mode: using subset of {TRAIN_SUBSET_SIZE} train, {VAL_SUBSET_SIZE} val/test")


# ============================================================
# DATASET LOADING & PREPROCESSING
# ============================================================
def load_squad_data(filepath):
    """Load SQuAD-format JSON dataset."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def squad_to_examples(squad_data, max_examples=None):
    """Convert SQuAD format to list of flat examples."""
    examples = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                # Handle None answers (common in test sets)
                answers = qa.get("answers")
                if answers is None:
                    answers = {"text": [], "answer_start": []}
                elif not isinstance(answers, dict):
                    answers = {"text": [], "answer_start": []}
                elif "text" not in answers:
                    answers["text"] = []
                    answers["answer_start"] = []

                example = {
                    "id": qa["id"],
                    "title": article.get("title", ""),
                    "context": context,
                    "question": qa["question"],
                    "is_impossible": qa.get("is_impossible", False),
                    "answers": answers,
                }
                examples.append(example)
                if max_examples and len(examples) >= max_examples:
                    return examples
    return examples


def preprocess_function(examples, tokenizer, max_length=MAX_LENGTH, doc_stride=DOC_STRIDE):
    """Tokenize and preprocess examples for extractive QA with PhoBERT.

    PhoBERT tokenizer (slow) doesn't support return_offsets_mapping or token_type_ids.
    We manually compute answer positions by token matching.
    """
    input_ids_list = []
    attention_mask_list = []
    start_positions_list = []
    end_positions_list = []

    sep_token_id = tokenizer.sep_token_id

    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        context = examples["context"][i]
        is_impossible = examples["is_impossible"][i]
        answers = examples["answers"][i]

        # Tokenize question + context with truncation and stride
        tokenized = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Handle multiple features (overflow)
        if isinstance(input_ids[0], list):
            num_features = len(input_ids)
        else:
            num_features = 1
            input_ids = [input_ids]
            attention_mask = [attention_mask]

        for feat_idx in range(num_features):
            ids = input_ids[feat_idx]
            mask = attention_mask[feat_idx]

            # Find context token positions: PhoBERT format is <s> Q </s></s> C </s>
            sep_indices = [j for j in range(len(ids)) if ids[j] == sep_token_id]

            if len(sep_indices) >= 2:
                context_start = sep_indices[1] + 1
                context_end = sep_indices[-1] - 1
            else:
                context_start = 0
                context_end = len(ids) - 1

            # Handle impossible questions or missing answers
            answer_texts = answers.get("text", []) if answers else []
            answer_starts = answers.get("answer_start", []) if answers else []

            if is_impossible or not answer_texts or not answer_starts:
                start_positions_list.append(0)
                end_positions_list.append(0)
                input_ids_list.append(ids)
                attention_mask_list.append(mask)
                continue

            answer_text = answer_texts[0]

            # Tokenize context and answer to find token-level positions
            context_tokens = tokenizer.tokenize(context)
            answer_tokens = tokenizer.tokenize(answer_text)

            # Find answer tokens in context tokens
            start_token_in_ctx = None
            end_token_in_ctx = None

            if len(answer_tokens) > 0 and len(answer_tokens) <= len(context_tokens):
                for j in range(len(context_tokens) - len(answer_tokens) + 1):
                    if context_tokens[j:j + len(answer_tokens)] == answer_tokens:
                        start_token_in_ctx = j
                        end_token_in_ctx = j + len(answer_tokens) - 1
                        break

            if start_token_in_ctx is not None:
                start_token = context_start + start_token_in_ctx
                end_token = context_start + end_token_in_ctx
                start_token = max(0, min(start_token, len(ids) - 1))
                end_token = max(0, min(end_token, len(ids) - 1))
            else:
                start_token = 0
                end_token = 0

            start_positions_list.append(start_token)
            end_positions_list.append(end_token)
            input_ids_list.append(ids)
            attention_mask_list.append(mask)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "start_positions": start_positions_list,
        "end_positions": end_positions_list,
    }


def prepare_dataset(squad_filepath, tokenizer, max_length=MAX_LENGTH, doc_stride=DOC_STRIDE,
                    max_examples=None):
    """Load, convert, and preprocess a SQuAD dataset split."""
    print(f"[INFO] Loading dataset from {squad_filepath}...")
    squad_data = load_squad_data(squad_filepath)
    examples = squad_to_examples(squad_data, max_examples=max_examples)
    num_qas = len(examples)
    print(f"[INFO] Loaded {num_qas} QA examples")

    dataset = Dataset.from_dict({
        "id": [e["id"] for e in examples],
        "title": [e["title"] for e in examples],
        "context": [e["context"] for e in examples],
        "question": [e["question"] for e in examples],
        "is_impossible": [e["is_impossible"] for e in examples],
        "answers": [e["answers"] for e in examples],
    })

    print(f"[INFO] Tokenizing {num_qas} examples...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length, doc_stride),
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


def examples_to_dataset(examples):
    """Convert list of examples to HuggingFace Dataset."""
    return Dataset.from_dict({
        "id": [e["id"] for e in examples],
        "title": [e["title"] for e in examples],
        "context": [e["context"] for e in examples],
        "question": [e["question"] for e in examples],
        "is_impossible": [e["is_impossible"] for e in examples],
        "answers": [e["answers"] for e in examples],
    })


# ============================================================
# CUSTOM TRAINER FOR LOSS & METRIC TRACKING
# ============================================================
class QATrainer(Trainer):
    """Custom Trainer that logs training loss."""

    def __init__(self, *args, **kwargs):
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_em": [],
            "val_f1": [],
        }
        super().__init__(*args, **kwargs)

    def log(self, logs):
        super().log(logs)
        if "loss" in logs:
            self.training_history["train_loss"].append(logs["loss"])


# ============================================================
# VALIDATION & TEST EVALUATION
# ============================================================
def evaluate_model(model, tokenizer, dataset, device, max_length=MAX_LENGTH):
    """Run inference and compute EM/F1 scores."""
    model.eval()

    references = {}
    raw_examples = list(dataset)
    for example in raw_examples:
        qid = example["id"]
        answers = example.get("answers", {})
        if isinstance(answers, dict):
            gt_answers = answers.get("text", [])
        else:
            gt_answers = []
        references[qid] = gt_answers if gt_answers else []

    predictions = {}
    print(f"[INFO] Running evaluation on {len(raw_examples)} examples...")

    for example in tqdm(raw_examples, desc="Evaluating"):
        qid = example["id"]
        question = example["question"]
        context = example["context"]

        inputs = tokenizer(
            question,
            context,
            max_length=max_length,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()

            if end_idx < start_idx:
                end_idx = min(start_idx + 5, input_ids.shape[1] - 1)

            start_idx = max(0, min(start_idx, input_ids.shape[1] - 1))
            end_idx = max(0, min(end_idx, input_ids.shape[1] - 1))

            answer_ids = input_ids[0][start_idx:end_idx + 1].tolist()
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(answer_ids)
            )
            answer = answer.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()

        predictions[qid] = answer if answer else "Không tìm thấy câu trả lời."

    # Compute metrics only if we have ground truth answers
    evalable_count = sum(1 for v in references.values() if v)
    if evalable_count == 0:
        print("[WARN] No ground truth answers available - returning raw predictions")
        model.train()
        return {"exact_match": 0.0, "f1": 0.0, "note": "No ground truth in test set"}, predictions, references

    total_em = 0.0
    total_f1 = 0.0
    count = 0

    for qid, pred in predictions.items():
        gts = references.get(qid, [])
        if not gts:
            continue
        em = max(compute_exact_match(pred, gt) for gt in gts)
        f1 = max(compute_f1(pred, gt) for gt in gts)
        total_em += em
        total_f1 += f1
        count += 1

    em_score = 100.0 * total_em / count if count > 0 else 0.0
    f1_score = 100.0 * total_f1 / count if count > 0 else 0.0

    model.train()
    return {"exact_match": em_score, "f1": f1_score}, predictions, references


# ============================================================
# VISUALIZATION
# ============================================================
def plot_training_curves(history, save_path):
    """Plot training loss, validation metrics, and EM/F1 curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Training loss
    if history["train_loss"]:
        steps = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(steps, history["train_loss"], 'b-', alpha=0.3, label='Training Loss (per step)')
        if len(history["train_loss"]) > 10:
            window = min(10, len(history["train_loss"]) // 2)
            smooth = np.convolve(history["train_loss"], np.ones(window)/window, mode='valid')
            axes[0].plot(range(window, len(history["train_loss"]) + 1), smooth, 'b-', linewidth=2, label=f'Smoothed (window={window})')
        axes[0].set_title('Training Loss Curve')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Plot 2: Validation EM & F1
    if history["val_em"] and history["val_f1"]:
        epochs = range(1, len(history["val_em"]) + 1)
        axes[1].plot(epochs, history["val_em"], 'r-', marker='o', label='Exact Match', linewidth=2)
        axes[1].plot(epochs, history["val_f1"], 'g-', marker='s', label='F1-Score', linewidth=2)
        axes[1].set_title('Validation EM & F1 Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)

    # Plot 3: Summary
    if history["train_loss"]:
        axes[2].text(0.1, 0.5,
                     f'Training completed\n'
                     f'Total steps: {len(history["train_loss"])}\n'
                     f'Final loss: {history["train_loss"][-1]:.4f}\n'
                     f'Initial loss: {history["train_loss"][0]:.4f}',
                     fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title('Training Summary')
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


def plot_em_f1_comparison(val_metrics, test_metrics, save_path):
    """Compare validation and test EM/F1 scores."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Validation', 'Test (proxy)']
    em_scores = [val_metrics['exact_match'], test_metrics['exact_match']]
    f1_scores = [val_metrics['f1'], test_metrics['f1']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, em_scores, width, label='Exact Match', color='red', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='green', alpha=0.8)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('PhoBERT QA Performance: Validation vs Test', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] EM/F1 comparison saved to {save_path}")


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================
def main():
    print("=" * 70)
    print("PhoBERT QA Training on UIT-ViQuAD 2.0")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Load tokenizer
    print("\n[STEP 1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] Tokenizer loaded: {MODEL_NAME}")
    print(f"[INFO] Vocab size: {tokenizer.vocab_size}")

    # Determine dataset sizes
    if USE_FULL_DATASET:
        train_max = None
        val_max = None
        test_max = None
    else:
        train_max = TRAIN_SUBSET_SIZE
        val_max = VAL_SUBSET_SIZE
        test_max = VAL_SUBSET_SIZE

    # Dataset info
    train_qas = sum(len(p['qas']) for a in load_squad_data(TRAIN_FILE)['data'] for p in a['paragraphs'])
    val_qas = sum(len(p['qas']) for a in load_squad_data(VAL_FILE)['data'] for p in a['paragraphs'])
    test_qas = sum(len(p['qas']) for a in load_squad_data(TEST_FILE)['data'] for p in a['paragraphs'])
    print(f"[INFO] Full dataset - Train: {train_qas}, Val: {val_qas}, Test: {test_qas}")

    # Tokenize datasets
    print("\n[STEP 2] Loading and preprocessing datasets...")

    print("[INFO] Tokenizing train dataset...")
    train_dataset = prepare_dataset(TRAIN_FILE, tokenizer, max_examples=train_max)
    print(f"[INFO] Train dataset: {len(train_dataset)} features")

    print("[INFO] Tokenizing validation dataset...")
    val_dataset = prepare_dataset(VAL_FILE, tokenizer, max_examples=val_max)
    print(f"[INFO] Validation dataset: {len(val_dataset)} features")

    print("[INFO] Tokenizing test dataset...")
    test_dataset = prepare_dataset(TEST_FILE, tokenizer, max_examples=test_max)
    print(f"[INFO] Test dataset: {len(test_dataset)} features")

    # Load model
    print("\n[STEP 3] Loading PhoBERT model...")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    print(f"[INFO] Model loaded: {MODEL_NAME}")
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training arguments
    print("\n[STEP 4] Setting up training configuration...")
    
    # Mixed precision: FP16 works on both CUDA and ROCm
    # AMD ROCm uses HIP FP16 (half precision) - compatible with PyTorch AMP
    use_fp16 = IS_GPU  # Both CUDA and ROCm support FP16
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_dir=LOGS_DIR,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=use_fp16,
        bf16=False,  # AMD ROCm doesn't support BF16 on RDNA2
        dataloader_num_workers=0,
        dataloader_pin_memory=IS_GPU,
        remove_unused_columns=False,
        # ROCm optimization: use HIP graphs for repeated kernel launches
        **({"torch_compile": True} if IS_ROCM else {}),
    )

    print(f"[INFO] Training configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  FP16: {IS_GPU and training_args.fp16}")
    print(f"  ROCm (AMD): {IS_ROCM}")
    print(f"  GPU: {GPU_NAME}")
    print(f"  Warmup ratio: {WARMUP_RATIO}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Device: {DEVICE}")

    # Prepare raw datasets for evaluation
    val_raw_examples = squad_to_examples(load_squad_data(VAL_FILE), max_examples=val_max)
    val_raw_dataset = examples_to_dataset(val_raw_examples)

    # Initialize Trainer
    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Validation callback for epoch-end evaluation
    from transformers import TrainerCallback

    class ValidationCallback(TrainerCallback):
        def __init__(self, trainer, val_dataset_raw, tokenizer, device):
            self.trainer = trainer
            self.val_dataset_raw = val_dataset_raw
            self.tokenizer = tokenizer
            self.device = device

        def on_epoch_end(self, args, state, control, **kwargs):
            print("\n[INFO] Running validation evaluation...")
            metrics, predictions, references = evaluate_model(
                self.trainer.model, self.tokenizer, self.val_dataset_raw, self.device
            )
            print(f"[INFO] Validation EM: {metrics['exact_match']:.2f}%")
            print(f"[INFO] Validation F1: {metrics['f1']:.2f}%")

            self.trainer.training_history["val_em"].append(metrics["exact_match"])
            self.trainer.training_history["val_f1"].append(metrics["f1"])

            save_path = os.path.join(VIZ_DIR, "val_predictions_sample.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "metrics": metrics,
                    "sample_predictions": dict(list(predictions.items())[:30]),
                    "sample_ground_truths": dict(list(references.items())[:30]),
                }, f, ensure_ascii=False, indent=2)

    val_callback = ValidationCallback(trainer, val_raw_dataset, tokenizer, DEVICE)
    trainer.add_callback(val_callback)

    # Start training
    print("\n[STEP 5] Starting training...")
    print("=" * 70)
    start_time = time.time()

    train_result = trainer.train()
    training_time = time.time() - start_time

    print(f"\n[INFO] Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"[INFO] Training loss: {train_result.metrics.get('train_loss', 'N/A')}")

    # Save training history
    print("\n[STEP 6] Saving training history...")
    history_path = os.path.join(VIZ_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.training_history, f, indent=2, default=str)
    print(f"[INFO] Training history saved to {history_path}")

    # Plot training curves
    print("\n[STEP 7] Generating visualizations...")
    curves_path = os.path.join(VIZ_DIR, "training_curves.png")
    plot_training_curves(trainer.training_history, curves_path)

    # Final validation evaluation
    print("\n[STEP 8] Running final validation evaluation...")
    val_metrics, val_predictions, val_references = evaluate_model(
        model, tokenizer, val_raw_dataset, DEVICE
    )
    print(f"[INFO] Final Validation EM: {val_metrics['exact_match']:.2f}%")
    print(f"[INFO] Final Validation F1: {val_metrics['f1']:.2f}%")

    # Test evaluation
    print("\n[STEP 9] Running test evaluation...")
    test_raw_examples = squad_to_examples(load_squad_data(TEST_FILE), max_examples=test_max)
    test_raw_dataset = examples_to_dataset(test_raw_examples)
    test_metrics, test_predictions, test_references = evaluate_model(
        model, tokenizer, test_raw_dataset, DEVICE
    )
    print(f"[INFO] Test Exact Match: {test_metrics['exact_match']:.2f}%")
    print(f"[INFO] Test F1 Score: {test_metrics['f1']:.2f}%")

    # Generate comparison plot
    comparison_path = os.path.join(VIZ_DIR, "em_f1_comparison.png")
    plot_em_f1_comparison(val_metrics, test_metrics, comparison_path)

    # Save model and tokenizer
    print("\n[STEP 10] Saving model and tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Model saved to {OUTPUT_DIR}")

    # Save test predictions
    test_pred_path = os.path.join(VIZ_DIR, "test_predictions.json")
    with open(test_pred_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": test_metrics,
            "predictions": test_predictions,
            "ground_truths": test_references,
        }, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Test predictions saved to {test_pred_path}")

    # Generate final report
    print("\n[STEP 11] Generating final report...")
    report_path = os.path.join(PROJECT_DIR, "training_report.json")
    report = {
        "model_name": MODEL_NAME,
        "training_config": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "num_epochs": NUM_EPOCHS,
            "max_length": MAX_LENGTH,
            "fp16_enabled": IS_GPU,
            "is_rocm": IS_ROCM,
            "gpu_name": GPU_NAME,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "device": str(DEVICE),
        },
        "dataset_info": {
            "train_qa_pairs_full": train_qas,
            "val_qa_pairs_full": val_qas,
            "test_qa_pairs_full": test_qas,
            "train_qa_pairs_used": len(train_dataset),
            "val_qa_pairs_used": len(val_dataset),
            "test_qa_pairs_used": len(test_dataset),
            "use_full_dataset": USE_FULL_DATASET,
        },
        "training_history": trainer.training_history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_time_seconds": training_time,
        "output_dir": OUTPUT_DIR,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[INFO] Final report saved to {report_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"Validation EM: {val_metrics['exact_match']:.2f}%")
    print(f"Validation F1: {val_metrics['f1']:.2f}%")
    print(f"Test EM: {test_metrics['exact_match']:.2f}%")
    print(f"Test F1: {test_metrics['f1']:.2f}%")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Visualizations: {VIZ_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
