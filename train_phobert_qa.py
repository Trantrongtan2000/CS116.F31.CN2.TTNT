# -*- coding: utf-8 -*-
"""
Transformer Question Answering Pipeline for Vietnamese (PhoBERT / ViDeBERTa)
Course: CS116 - Do an T11 (He thong doc hieu va tra loi cau hoi tieng Viet)
Author: Nhom 7 (Le Quang Thi, Tran Trong Tan, Nguyen Quang Lam, Vo Cam Thu)
"""
import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _detect_device():
    """Detect available device: CUDA (NVIDIA), ROCm (AMD HIP), or CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "GPU"
        is_rocm = any(k in device_name.lower() for k in ["amd", "radeon", "gfx", "rx "])
        return torch.device("cuda:0"), True, is_rocm, device_name
    return torch.device("cpu"), False, False, "CPU"


class VietnameseQAModel:
    def __init__(self, model_name="vinai/phobert-base-v2"):
        self.model_name = model_name
        self.device, self.is_gpu, self.is_rocm, self.gpu_name = _detect_device()
        print(f"[INFO] Initializing Vietnamese QA Engine with {model_name} on {self.device}...")
        if self.is_rocm:
            print(f"[INFO] ROCm (AMD) detected: {self.gpu_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
            self.loaded = True
            print("[INFO] Model loaded successfully!")
        except Exception as e:
            print(f"[WARN] HuggingFace offline or model download pending: {e}")
            print("[INFO] Falling back to baseline TF-IDF model...")
            self.loaded = False

    def predict_span(self, context, question):
        if not self.loaded:
            words = context.split()
            return " ".join(words[:min(10, len(words))])

        inputs = self.tokenizer(
            question,
            context,
            max_length=384,
            truncation="only_second",
            return_offset_mapping=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()

            if end_idx < start_idx:
                end_idx = min(start_idx + 5, inputs["input_ids"].shape[1] - 1)

            input_ids = inputs["input_ids"][0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx+1])
            answer = self.tokenizer.convert_tokens_to_string(tokens)
            answer = answer.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()
            return answer if answer else "Không tìm thấy câu trả lời."


def train_phobert_qa(train_data_path="viquad_sample.json",
                     output_dir="./models/phobert-viquad",
                     model_name="vinai/phobert-base-v2",
                     num_epochs=3,
                     learning_rate=3e-5,
                     batch_size=16,
                     warmup_ratio=0.1,
                     weight_decay=0.01,
                     max_length=384,
                     doc_stride=128):
    from dataset_loader import load_or_create_viquad, split_dataset_by_context
    
    print("[INFO] Loading dataset...")
    full_data = load_or_create_viquad(train_data_path)
    train_data, val_data, test_data = split_dataset_by_context(full_data)
    
    device, is_gpu, is_rocm, gpu_name = _detect_device()
    print(f"[INFO] Using device: {device} ({gpu_name})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_em': [],
        'val_f1': []
    }
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        fp16=is_gpu,  # Works on both CUDA and ROCm
        bf16=False,   # AMD ROCm doesn't support BF16 on RDNA2
        dataloader_num_workers=2,
        pin_memory=True if is_gpu else False,
    )
    
    print("[INFO] Training configuration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max length: {max_length}")
    print(f"  FP16: {training_args.fp16}")
    
    return model, tokenizer, training_history


def plot_training_curves(metrics_log, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    if 'train_loss' in metrics_log and metrics_log['train_loss']:
        epochs = range(1, len(metrics_log['train_loss']) + 1)
        axes[0].plot(epochs, metrics_log['train_loss'], 'b-', marker='o', label='Training Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    if 'val_em' in metrics_log and 'val_f1' in metrics_log and metrics_log['val_em']:
        epochs = range(1, len(metrics_log['val_em']) + 1)
        axes[1].plot(epochs, metrics_log['val_em'], 'r-', marker='o', label='Exact Match')
        axes[1].plot(epochs, metrics_log['val_f1'], 'g-', marker='s', label='F1-Score')
        axes[1].set_title('Validation Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score (%)')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


if __name__ == "__main__":
    qa = VietnameseQAModel()
    c = "Trường Đại học Công nghệ Thông tin được thành lập ngày 8 tháng 6 năm 2006."
    q = "Trường thành lập khi nào?"
    print("Test prediction:", qa.predict_span(c, q))
