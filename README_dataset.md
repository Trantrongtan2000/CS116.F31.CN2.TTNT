# UIT-ViQuAD 2.0 Integration Guide — CS116 MRC Project

## Tổng quan

Dự án CS116 (Đồ án T11) sử dụng **UIT-ViQuAD 2.0** — bộ dữ liệu tiếng Việt cho bài toán
**Extractive Machine Reading Comprehension (MRC)**.

| Thành phần | Mô tả |
| :--- | :--- |
| **Nguồn dữ liệu** | `taidng/UIT-ViQuAD2.0` (HuggingFace Hub) |
| **Định dạng** | SQuAD 2.0 (có `is_impossible`, `plausible_answers`) |
| **Số lượng QA** | 39,569 câu hỏi (Train: 28,454 | Val: 3,814 | Test: 7,301) |
| **Số contexts** | 6,399 contexts (Train: 4,101 | Val: 557 | Test: 1,241) |

## Cấu trúc file

```
01_CS116_DoAn_T11_MRC/
├── viquad_sample.json          # Mẫu nhỏ (4 QA) — giữ nguyên để test nhanh
├── viquad2_train.json          # Full train set (SQuAD format)
├── viquad2_validation.json     # Full validation set (SQuAD format)
├── viquad2_test.json           # Full test set (SQuAD format)
├── viquad2_split_stats.json    # Thống kê chi tiết từng split
├── dataset_loader.py           # Loader hỗ trợ tải từ local + HuggingFace
└── README_dataset.md           # File hướng dẫn này
```

## Cách sử dụng

### 1. Tải toàn bộ dataset về local

```python
from datasets import load_dataset
import json, os

project_dir = os.path.dirname(__file__)  # thư mục chứa dataset_loader.py

ds = load_dataset("taidng/UIT-ViQuAD2.0")
for split in ["train", "validation", "test"]:
    ds[split].to_json(os.path.join(project_dir, f"viquad2_{split}.json"))
```

### 2. Load dataset bằng `dataset_loader.py`

```python
from dataset_loader import load_viquad2_0

# Load từng split (tự động ưu tiên local file, fallback về HuggingFace)
train_data = load_viquad2_0(split="train")      # 28,454 QA
val_data   = load_viquad2_0(split="validation") # 3,814 QA
test_data  = load_viquad2_0(split="test")       # 7,301 QA
```

### 3. Dùng với PhoBERT

```python
from dataset_loader import load_viquad2_0, preprocess_for_phobert
from transformers import AutoTokenizer

train_data = load_viquad2_0("train")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
processed = preprocess_for_phobert(train_data, tokenizer)
```

### 4. Chuyển đổi sang HuggingFace Dataset format

Nếu cần dùng trực tiếp với HuggingFace Trainer:

```python
from datasets import Dataset

train_data = load_viquad2_0("train")
# Flatten SQuAD format -> HuggingFace Dataset rows
rows = []
for article in train_data["data"]:
    for para in article["paragraphs"]:
        for qa in para["qas"]:
            rows.append({
                "context": para["context"],
                "question": qa["question"],
                "answers": qa["answers"],
                "id": qa["id"],
            })
hf_ds = Dataset.from_dict({"data": rows})
```

## Context-level Split & Data Leakage

Theo yêu cầu của giảng viên **ThS. Nguyễn Hữu Quyền**, dữ liệu phải được chia ở mức
**Context-level**: mỗi context (đoạn văn) chỉ được gán cho một split duy nhất để tránh
**data leakage**. Dataset `taidng/UIT-ViQuAD2.0` đã được chia theo nguyên tắc này.

### ⚠️ Lưu ý: Data leakage đã phát hiện

| Pair | Context overlap | Title overlap |
| :--- | :--- | :--- |
| Train ↔ Test | **449** contexts | 17 titles |
| Val ↔ Test | **133** contexts | 4 titles |
| Train ↔ Val | 0 contexts | 0 titles |

**Hành động khuyến nghị:**
- Áp dụng `deduplicate_contexts()` trong `dataset_loader.py` để loại bỏ context trùng lặp
  giữa train và test nếu accuracy là mụi tiêu hàng đầu.
- Hoặc giữ nguyên vì đây là đặặc thù của benchmark chính thức (có thể là intentional overlap).

## Thống kê chi tiết

| Split | QA pairs | Contexts | Titles | Impossible QA | Impossible % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 28,454 | 4,101 | 138 | 9,216 | 32.38% |
| Validation | 3,814 | 557 | 19 | 1,161 | 30.44% |
| Test | 7,301 | 1,241 | 48 | 0 | 0.00% |
| **Total** | **39,569** | **6,399** | **197** | **10,377** | — |

> **Lưu ý:** Test set không chứa câu hỏi impossible — đây là đặc thù của benchmark.

## Các file liên quan trong code

| File | Vai trò |
| :--- | :--- |
| `dataset_loader.py` | Load/preprocess dataset (SQuAD + HuggingFace format) |
| `train_phobert_qa.py` | Fine-tune PhoBERT/ViDeBERTa trên dataset |
| `baseline_model.py` | Mô hình baseline TF-IDF/BM25 |
| `eval_metrics.py` | Tính EM, F1 score |
| `app_streamlit.py` | Web demo tương tác |
