# -*- coding: utf-8 -*-
"""
Dataset Loader for UIT-ViQuAD 1.0/2.0 (Vietnamese Question Answering Dataset)
Course: CS116.F31.CN2.TTNT - Do an T11: He thong doc hieu va tra loi cau hoi tieng Viet
Nhom 7 (Dai dien: Le Quang Thi, Tran Trong Tan, Nguyen Quang Lam, Vo Cam Thu)
"""
import os
import json
from typing import Dict, List, Any
from datasets import Dataset, DatasetDict

def load_or_create_viquad(json_path="viquad_sample.json"):
    """
    Reads UIT-ViQuAD SQuAD-formatted JSON dataset or creates representative benchmark data.
    """
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Loaded UIT-ViQuAD dataset from {json_path}")
        return data

    print("[INFO] Creating representative UIT-ViQuAD benchmark samples...")
    sample_data = {
        "version": "1.0",
        "data": [
            {
                "title": "Trường Đại học Công nghệ Thông tin - ĐHQG-HCM",
                "paragraphs": [
                    {
                        "context": "Trường Đại học Công nghệ Thông tin là một trường đại học thành viên của Đại học Quốc gia Thành phố Hồ Chí Minh, được thành lập theo Quyết định số 134/2006/QĐ-TTg ngày 8 tháng 6 năm 2006 của Thủ tướng Chính phủ. Trường có nhiệm vụ đào tạo nguồn nhân lực công nghệ thông tin chất lượng cao, đóng góp tích cực vào sự phát triển của nền công nghiệp công nghệ thông tin Việt Nam.",
                        "qas": [
                            {
                                "id": "uit_001",
                                "question": "Trường Đại học Công nghệ Thông tin được thành lập vào ngày tháng năm nào?",
                                "answers": [{"text": "ngày 8 tháng 6 năm 2006", "answer_start": 140}]
                            },
                            {
                                "id": "uit_002",
                                "question": "Trường Đại học Công nghệ Thông tin là thành viên của hệ thống đại học nào?",
                                "answers": [{"text": "Đại học Quốc gia Thành phố Hồ Chí Minh", "answer_start": 71}]
                            }
                        ]
                    },
                    {
                        "context": "Trí tuệ nhân tạo là một ngành thuộc lĩnh vực khoa học máy tính nhằm tạo ra những hệ thống hoặc máy móc có khả năng mô phỏng các quá trình trí tuệ của con người. Các quá trình này bao gồm việc học tập, lập luận, tự sửa lỗi và xử lý ngôn ngữ tự nhiên.",
                        "qas": [
                            {
                                "id": "ai_001",
                                "question": "Trí tuệ nhân tạo là ngành thuộc lĩnh vực nào?",
                                "answers": [{"text": "khoa học máy tính", "answer_start": 44}]
                            },
                            {
                                "id": "ai_002",
                                "question": "Các quá trình trí tuệ được mô phỏng bao gồm những gì?",
                                "answers": [{"text": "học tập, lập luận, tự sửa lỗi và xử lý ngôn ngữ tự nhiên", "answer_start": 190}]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    return sample_data


DATASET_NAME = "taidng/UIT-ViQuAD2.0"

def load_viquad2_0(split: str = "train") -> Dict[str, Any]:
    """
    Load UIT-ViQuAD 2.0 dataset from HuggingFace or local SQuAD-format JSON file.
    Returns SQuAD-formatted dictionary compatible with existing code.

    Args:
        split: One of 'train', 'validation', 'test'.

    Returns:
        SQuAD-formatted dict with 'version' and 'data' keys.
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split: '{split}'. Choose from {valid_splits}.")

    # --- Priority 1: Local SQuAD-format JSON file ---
    local_file = os.path.join(os.path.dirname(__file__), f"viquad2_{split}.json")
    if os.path.exists(local_file):
        with open(local_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Loaded local SQuAD-format file: {local_file}")
        return data

    # --- Priority 2: Download from HuggingFace Hub ---
    try:
        from datasets import load_dataset as hf_load_dataset

        print(f"[INFO] Loading UIT-ViQuAD 2.0 dataset '{DATASET_NAME}' from HuggingFace (split: {split})...")
        ds = hf_load_dataset(DATASET_NAME, split=split)

        squad_format = {"version": "2.0", "data": []}
        articles_by_title: Dict[str, Dict[str, Any]] = {}

        for item in ds:
            title = item.get("title", "unknown")
            context = item.get("context", "")
            question = item.get("question", "")
            answers = item.get("answers", {})
            qa_id = item.get("id", f"viquad_{hash(question + context) % 1000000}")
            is_impossible = item.get("is_impossible", False)

            if title not in articles_by_title:
                articles_by_title[title] = {
                    "title": title,
                    "paragraphs": []
                }

            # Find or create paragraph
            existing_paragraph = None
            for para in articles_by_title[title]["paragraphs"]:
                if para["context"] == context:
                    existing_paragraph = para
                    break

            if existing_paragraph is None:
                existing_paragraph = {
                    "context": context,
                    "qas": []
                }
                articles_by_title[title]["paragraphs"].append(existing_paragraph)

            # Build answers list
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            answer_starts = answers.get("answer_start", []) if isinstance(answers, dict) else []

            if isinstance(answer_texts, str):
                answer_texts = [answer_texts]
            if isinstance(answer_starts, int):
                answer_starts = [answer_starts]

            answer_list = []
            for t, s in zip(answer_texts or [""], answer_starts or [0]):
                answer_list.append({"text": t, "answer_start": s})

            if not answer_list:
                answer_list = [{"text": "", "answer_start": 0}]

            qa_entry = {
                "id": qa_id,
                "question": question,
                "answers": answer_list,
                "is_impossible": is_impossible,
            }

            if item.get("plausible_answers") is not None:
                qa_entry["plausible_answers"] = item["plausible_answers"]

            existing_paragraph["qas"].append(qa_entry)

        squad_format["data"] = list(articles_by_title.values())

        num_qas = sum(len(p["qas"]) for a in squad_format["data"] for p in a["paragraphs"])
        print(f"[INFO] Loaded and converted {num_qas} QA pairs from UIT-ViQuAD 2.0")
        return squad_format

    except Exception as e:
        print(f"[WARN] Failed to load from HuggingFace: {e}")
        print("[INFO] Falling back to local sample dataset...")
        return load_or_create_viquad()


def load_viquad_from_huggingface(split: str = "train") -> Dict[str, Any]:
    """
    Alias for load_viquad2_0() to maintain backward compatibility.
    """
    return load_viquad2_0(split=split)


def deduplicate_contexts(
    train_data: Dict, val_data: Dict, test_data: Dict
) -> tuple:
    """
    Loại bỏ context trùng lặp giữa train/val/test để tránh data leakage.
    Áp dụng nguyên tắc Context-level split: mỗi context chỉ xuất hiện ở một split.

    Args:
        train_data, val_data, test_data: SQuAD-format dicts (output of load_viquad2_0)

    Returns:
        Tuple (train_data, val_data, test_data) — đã được deduplicated
    """
    def extract_contexts(data: Dict) -> set:
        return {p["context"] for a in data["data"] for p in a["paragraphs"]}

    train_ctx = extract_contexts(train_data)
    val_ctx = extract_contexts(val_data)
    test_ctx = extract_contexts(test_data)

    # Context chỉ giữ ở split đầu tiên (train > val > test)
    val_unique = val_ctx - train_ctx - test_ctx
    test_unique = test_ctx - train_ctx - val_ctx

    def filter_by_contexts(data: Dict, allowed: set) -> Dict:
        """Giữ lại paragraphs có context nằm trong allowed set."""
        filtered_data = []
        for article in data["data"]:
            kept_paragraphs = [
                p for p in article["paragraphs"]
                if p["context"] in allowed
            ]
            if kept_paragraphs:
                filtered_data.append({
                    "title": article["title"],
                    "paragraphs": kept_paragraphs
                })
        return {"version": data.get("version", "2.0"), "data": filtered_data}

    # Train giữ nguyên (ưu tiên cao nhất)
    filtered_train = filter_by_contexts(train_data, train_ctx)

    # Val chỉ giữ context không bị trùng với train hoặc test
    filtered_val = filter_by_contexts(val_data, val_unique)

    # Test chỉ giữ context không bị trùng với train hoặc val
    filtered_test = filter_by_contexts(test_data, test_unique)

    # Thống kê
    orig_train = len(train_ctx)
    orig_val = len(val_ctx)
    orig_test = len(test_ctx)
    new_train = len(extract_contexts(filtered_train))
    new_val = len(extract_contexts(filtered_val))
    new_test = len(extract_contexts(filtered_test))

    print(f"[DEDUP] Train contexts: {orig_train} -> {new_train}")
    print(f"[DEDUP] Val contexts:   {orig_val} -> {new_val}")
    print(f"[DEDUP] Test contexts:  {orig_test} -> {new_test}")

    return filtered_train, filtered_val, filtered_test


def split_dataset_by_context(data: Dict, train_ratio: float = 0.7, dev_ratio: float = 0.15):
    """
    Split dataset by context level to prevent data leakage.
    All questions from the same context must belong to the same split.
    """
    import random
    random.seed(42)
    
    contexts = [p for d in data["data"] for p in d["paragraphs"]]
    random.shuffle(contexts)
    
    n_total = len(contexts)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    
    train_contexts = contexts[:n_train]
    dev_contexts = contexts[n_train:n_train + n_dev]
    test_contexts = contexts[n_train + n_dev:]
    
    def build_split(context_list):
        return {
            "version": data.get("version", "1.0"),
            "data": [
                {
                    "title": "split",
                    "paragraphs": context_list
                }
            ]
        }
    
    return (
        build_split(train_contexts),
        build_split(dev_contexts),
        build_split(test_contexts)
    )


def preprocess_for_phobert(data: Dict, tokenizer, max_length: int = 384):
    """
    Preprocess SQuAD-formatted data for PhoBERT fine-tuning.
    Returns tokenized inputs suitable for transformers pipeline.
    """
    questions = []
    contexts = []
    answers = []
    
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                questions.append(qa["question"])
                contexts.append(context)
                answers.append(qa["answers"][0]["text"] if qa["answers"] else "")
    
    print(f"[INFO] Preprocessing {len(questions)} Q&A pairs for PhoBERT...")
    
    # Tokenize - this would be called in the training script
    return {
        "questions": questions,
        "contexts": contexts,
        "answers": answers
    }


if __name__ == "__main__":
    # Test loading local sample dataset
    data = load_or_create_viquad()
    print("Local dataset samples:", len(data["data"]))

    # Test loading UIT-ViQuAD 2.0 from local SQuAD-format files
    for split_name in ["train", "validation", "test"]:
        split_data = load_viquad2_0(split=split_name)
        num_qas = sum(len(p["qas"]) for art in split_data["data"] for p in art["paragraphs"])
        num_contexts = sum(len(art["paragraphs"]) for art in split_data["data"])
        num_impossible = sum(
            1 for art in split_data["data"]
            for p in art["paragraphs"]
            for qa in p["qas"]
            if qa.get("is_impossible", False)
        )
        print(f"\n{split_name}: {num_qas} QAs, {num_contexts} contexts, {num_impossible} impossible questions")

    # Test loading from HuggingFace (fallback if local files missing)
    # data_hf = load_viquad2_0(split="train")
    # print("Loaded from HF:", len(data_hf["data"]), "articles")

    # Test splitting (local sample)
    train, dev, test = split_dataset_by_context(data)
    print(f"\nLocal split - Train: {len(train['data'][0]['paragraphs'])} contexts")
    print(f"Local split - Dev: {len(dev['data'][0]['paragraphs'])} contexts")
    print(f"Local split - Test: {len(test['data'][0]['paragraphs'])} contexts")

    # Test deduplication on full UIT-ViQuAD 2.0
    print("\n=== Context-level Deduplication Test ===")
    train_full = load_viquad2_0("train")
    val_full = load_viquad2_0("validation")
    test_full = load_viquad2_0("test")

    train_dedup, val_dedup, test_dedup = deduplicate_contexts(train_full, val_full, test_full)

    # Verify no leakage
    train_c = {p["context"] for a in train_dedup["data"] for p in a["paragraphs"]}
    val_c = {p["context"] for a in val_dedup["data"] for p in a["paragraphs"]}
    test_c = {p["context"] for a in test_dedup["data"] for p in a["paragraphs"]}
    assert len(train_c & val_c) == 0, "Still overlap between train and val!"
    assert len(train_c & test_c) == 0, "Still overlap between train and test!"
    assert len(val_c & test_c) == 0, "Still overlap between val and test!"
    print("[PASS] No data leakage detected after deduplication!")

    # Print final counts
    for name, d in [("Train", train_dedup), ("Val", val_dedup), ("Test", test_dedup)]:
        qas = sum(len(p["qas"]) for a in d["data"] for p in a["paragraphs"])
        ctxs = sum(len(a["paragraphs"]) for a in d["data"])
        print(f"  {name}: {qas} QAs, {ctxs} contexts")