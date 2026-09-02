# -*- coding: utf-8 -*-
"""Tiny CPU fine-tune — produces a GENUINE training curve for the report.

This is NOT the main result path (that is inference-only, see run_eval.py). Its
sole purpose is to satisfy the "loss / EM-F1 per epoch" visualisation requirement
with real, produced-not-invented data, within CPU limits. It therefore uses a
small model, a small subset, and a few epochs. Metrics here are expected to be
low — the honest curve is the point, not the score.

Model: distilbert-base-multilingual-cased (a compact multilingual encoder with a
QA head; far lighter than XLM-R for CPU). Uses a fast tokenizer with offset
mapping to compute correct start/end token labels — the alignment step the old
repo did unsafely.

Outputs:
  results/training_curve.json : {epoch, train_loss, val_em, val_f1}[]
"""
import argparse
import json
import os
import time

from mrc import data, metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_MODEL = "distilbert-base-multilingual-cased"


def prepare_features(examples, tokenizer, max_len, stride):
    """Tokenize (question, context) with offsets; compute start/end token labels."""
    import torch
    questions = [e.question.strip() for e in examples]
    contexts = [e.context for e in examples]
    enc = tokenizer(
        questions, contexts,
        truncation="only_second", max_length=max_len, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    offset_map = enc.pop("offset_mapping")

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_map):
        sample_idx = sample_map[i]
        ex = examples[sample_idx]
        seq_ids = enc.sequence_ids(i)
        # context token span within this feature
        ctx_start = seq_ids.index(1) if 1 in seq_ids else 0
        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1) if 1 in seq_ids else len(seq_ids) - 1

        if not ex.has_answer:
            start_positions.append(0)   # CLS -> "no answer"
            end_positions.append(0)
            continue
        ans = ex.answers[0]
        start_char = ex.answer_starts[0] if ex.answer_starts else ex.context.find(ans)
        end_char = start_char + len(ans)
        if start_char < 0 or offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
            continue
        tok_start = ctx_start
        while tok_start <= ctx_end and offsets[tok_start][0] <= start_char:
            tok_start += 1
        tok_end = ctx_end
        while tok_end >= ctx_start and offsets[tok_end][1] >= end_char:
            tok_end -= 1
        start_positions.append(max(ctx_start, tok_start - 1))
        end_positions.append(min(ctx_end, tok_end + 1))

    features = {
        "input_ids": torch.tensor(enc["input_ids"]),
        "attention_mask": torch.tensor(enc["attention_mask"]),
        "start_positions": torch.tensor(start_positions),
        "end_positions": torch.tensor(end_positions),
    }
    return features


def evaluate_epoch(model, tokenizer, val_examples, max_len, device):
    import torch
    model.eval()
    preds, refs = {}, data.references_dict(val_examples)
    with torch.no_grad():
        for ex in val_examples:
            enc = tokenizer(ex.question, ex.context, truncation="only_second",
                            max_length=max_len, padding="max_length",
                            return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            seq_ids = enc.sequence_ids(0)
            out = model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
            s = int(out.start_logits.argmax()); e = int(out.end_logits.argmax())
            if e < s or seq_ids[s] != 1 or seq_ids[e] != 1:
                preds[ex.id] = ""
            else:
                preds[ex.id] = ex.context[offsets[s][0]:offsets[e][1]].strip()
    model.train()
    return metrics.evaluate(preds, refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=200)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cpu")
    print(f"[INFO] Tiny fine-tune: {args.model} | train={args.train_size} "
          f"val={args.val_size} epochs={args.epochs} (CPU)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)

    train_ex = data.subset([e for e in data.load_split("train") if e.has_answer], args.train_size)
    val_ex = data.subset(data.load_split("validation"), args.val_size)
    feats = prepare_features(train_ex, tokenizer, args.max_len, args.stride)
    n = feats["input_ids"].shape[0]
    print(f"[INFO] {n} training features prepared.")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    curve = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            optim.zero_grad()
            out = model(input_ids=feats["input_ids"][idx].to(device),
                        attention_mask=feats["attention_mask"][idx].to(device),
                        start_positions=feats["start_positions"][idx].to(device),
                        end_positions=feats["end_positions"][idx].to(device))
            out.loss.backward()
            optim.step()
            losses.append(out.loss.detach().item())
        train_loss = sum(losses) / len(losses)
        val = evaluate_epoch(model, tokenizer, val_ex, args.max_len, device)
        curve.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                      "val_em": round(val["EM"], 2), "val_f1": round(val["F1"], 2)})
        print(f"[EPOCH {epoch}] loss={train_loss:.4f}  val_EM={val['EM']:.2f}  "
              f"val_F1={val['F1']:.2f}  ({time.time() - t0:.0f}s)")

    out = os.path.join(RESULTS_DIR, "training_curve.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "config": vars(args), "curve": curve},
                  f, ensure_ascii=False, indent=2)
    print(f"[INFO] Training curve saved to {out}")


if __name__ == "__main__":
    main()
