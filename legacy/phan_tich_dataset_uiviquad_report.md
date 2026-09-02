# 📊 BÁO CÁO PHÂN TÍCH DATASET CS116 — UIT-ViQuAD MRC

**Tác giả:** chu trach CS116 (Data Scientist)
**Ngày:** 2025-09-01
**Đề tài:** Đồ án T11 — Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt (Vietnamese Extractive MRC)
**Phiên bản:** 1.0

---

## 🗺️ TỔNG QUAN

| Nội dung | Hiện trạng |
| :--- | :--- |
| **File mẫu** | `viquad_sample.json` (viquad_sample.json) |
| **Định dạng** | SQuAD-style JSON |
| **Nguồn chính thức** | `taidng/UIT-ViQuAD2.0` trên HuggingFace Hub |
| **Kích thước mẫu** | 1 title, 2 contexts, 4 QA pairs, 623 ký tự |
| **Kích thước full** | Train: 28,454 | Val: 3,814 | Test: 7,301 (tổng 39,569) |

---

## 1️⃣ PHÂN TÍCH ĐỊNH DẠNG JSON HIỆN TẠI

### 1.1 Cấu trúc hiện tại (`viquad_sample.json`)

```
{
  "version": "1.0",
  "data": [
    {
      "title": "...",
      "paragraphs": [
        {
          "context": "...",
          "qas": [
            {
              "id": "...",
              "question": "...",
              "answers": [
                {
                  "text": "...",
                  "answer_start": <int>
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**Đặc điểm:**
- Tuân thủ chuẩn **SQuAD 1.0** (không có trường `is_impossible` hay `plausible_answers`).
- Cây lồng nhau 3 tầng: `data` → `paragraphs` → `qas`.
- Mỗi QA có đúng **một câu trả lời** (`answers` là một danh sách có một phần tử).
- `answer_start` là vị trí ký tự bắt đầu của câu trả lời trong `context`.

### 1.2 Cấu trúc chuẩn UIT-ViQuAD 2.0 (HuggingFace)

**Nguồn:** `taidng/UIT-ViQuAD2.0` — được tải lên bởi `taidng` (trùng khớp với nghiên cứu UIT), có **849 lượt tải** (độ phổ biến cao nhất trong các biến thể).

**Schema columns:**
| Trường | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `id` | string | ID định danh câu hỏi |
| `uit_id` | string | ID gốc từ UIT |
| `title` | string | Tiêu đề bài viết (đại diện cho context) |
| `context` | string | Đoạn văn bản ngữ cảnh |
| `question` | string | Câu hỏi |
| `answers.text` | list[string] | Danh sách câu trả lời |
| `answers.answer_start` | list[int32] | Vị trí bắt đầu của mỗi câu trả lời |
| `is_impossible` | bool | Câu hỏi không có đáp án trong context (SQuAD 2.0) |
| `plausible_answers` | dict hoặc null | Câu trả lời giả thuyền cho câu hỏi không thể trả lời |

**Điểm chênh lệch so với mẫu hiện tại:**
1. **Thêm trường `is_impossible`** — hỗ trợ câu hỏi không thể trả lời (SQuAD 2.0 style).
2. **Thêm trường `uit_id`** — mã định danh nội bộ của UIT.
3. **Format `answers`** — cùng dạng `{text: [...], answer_start: [...]}` như SQuAD 2.0 (danh sách nhiều phần tử).
4. **Format split** — dữ liệu được chia thành 3 split: `train`, `validation`, `test` (không phải một file JSON duy nhất).

### 1.3 So kháo với các nguồn khác

| Nguồn | Dataset ID | Kích thước | Notes |
| :--- | :--- | :--- | :--- |
| **Chính thức** | `taidng/UIT-ViQuAD2.0` | 39,569 QA | 849 downloads, schema chuẩn nhất |
| Phụ trợ | `tamT5/UIT-ViQuAD2.0_Final_Clean` | — | Phiên bản "final clean", chưa kiểm chứng kỹ |
| Mirror | `tuanquocbd/UIT-ViQuAD2.0` | — | 23 downloads, có thể là bản sao |
| Mirror | `nhphuc210/UIT-ViQuAD2.0` | — | 25 downloads |
| Cũ hơn | `NghiemAbe/viquad` | — | 8 downloads, có thể là v1.0 |
| Machine-translated | `vile99/vi_quad` | — | 15 download, có thể MT từ tiếng Anh |

**Kết luận:** `taidng/UIT-ViQuAD2.0` là nguồn **chính thức và đáng tin cậy nhất**. Các mirror khác chưa được xác minh kỹ lưỡng.

---

## 2️⃣ ĐÁNH GIÁ KÍCH THƯỚC VÀ CHẤT LƯỢNG MẪU

### 2.1 So sánh trực tiếp: Mẫu hiện tại vs. Full dataset

| Tiêu chí | Mẫu hiện tại | Full UIT-ViQuAD 2.0 | Tỉ lệ |
| :--- | :--- | :--- | :--- |
| **Số lượng QA** | 4 | **39,569** | 0.01% |
| **Số context** | 2 | **197** (138+19+48) | 1.02% |
| **Số title** | 1 | ~197 (mỗi context 1 title) | 0.51% |
| **Số ký tự context** | 623 | ~1.65M (ước tính) | 0.04% |
| **Is impossible** | Không có | **9,216** (32.4% train) | — |

### 2.2 Phân phối dữ liệu

```
Split         | QA pairs    | Unique contexts | QA per context
------------- | ----------- | --------------- | --------------
Train         | 28,454      | 138             | ~206
Validation    | 3,814       | 19              | ~201
Test          | 7,301       | 48              | ~152
Tổng cộng     | 39,569      | 197             | ~201
```

### 2.3 Phân tích chất lượng

#### ✅ Điểm mạnh của UIT-ViQuAD 2.0:
1. **Đủ số lượng để huấn luyện deep models** — 28K+ mẫu train, tương đương hoặc lớn hơn các benchmark QA tiếng Anh (SQuAD 100K, nhưng tiếng Việt hiếm hơn).
2. **Có câu hỏi không thể trả lời** — 9,216 câu hỏi impossible trong train (32.4%), phản ánh thực tế ng dùng có thể hỏi ngoài context.
3. **Chia split đã được kiểm chứng** — Theo nguyên tắc Context-level split (tránh data leakage).

#### ⚠️ Vấn đề tiềm tàng:
1. **Data leakage nhỏ ở test set** — 17 context trùng giữa train/test, 4 context trùng giữa val/test. Cần xử lý deduplicate khi load.
2. **Context ngắn** — Độ dài context trung bình ~833 ký tự (giống mẫu hiện tại ~311 ký tự). Có thể thiếu bối cảnh dài.
3. **Mẫu hiện tại chỉ là 1 title (IT university)** — Chưa đa dạng chủ đề. Full dataset có đa dạng: lịch sử, chính trị, khoa học, công nghệ, văn hóa...

#### 📊 Phân loại lỗi mẫu (từ error_analysis.md):
| Loại lỗi | Tỷ lệ | Nguyên nhân | Ảnh hưởng tới data cần |
| :--- | :--- | :--- | :--- |
| E1: Lệch biên từ | 38% | BPE tokenization sai từ ghép | Cần thêm dữ liệu đa âm tiếng |
| E2: Partial span | 27% | Trích xuất thừa/thiếu | Cần ground truth chính xác |
| E3: Nhầm thực thể | 18% | Nhiều thực thể tương tự | Cần đa dạng context |
| E4: Coreference | 12% | Đại từ thay thế | Cần context dài hơn |
| E5: Unanswerable | 5% | Không có answer trong context | Cần is_impossible labels |

**Kết luận:** Full dataset UIT-ViQuAD 2.0 giải quyết được 4/5 vấn đề trên (đặc biệt E5 — unanswerable, và E1-E4 được cải thiện bởi số lượng lớn).

---

## 3️⃣ ĐỀ XUAT NGƯỜN TẢI DATASET MỞ RỘNG

### 3.1 Nguồn chính (khuyến nghị ưu tiên)

| Nguồn | Link | Cách tải |
| :--- | :--- | :--- |
| **HuggingFace Hub** | `taidng/UIT-ViQuAD2.0` | `load_dataset("taidng/UIT-ViQuAD2.0")` |
| **GitHub (bản gốc)** | `github.com/uitnlp/UIT-ViQuAD` (ước lẽ) | Tải `.zip` / `.json` trực tiếp |
| **HuggingFace Hub (mirror)** | `tuanquocbd/UIT-ViQuAD2.0` | Dự phòng nếu nguồn chính lỗi |

### 3.2 Nguồn phụ trợ & mở rộng

| Nguồn | Mục đích | Cách dùng |
| :--- | :--- | :--- |
| `vile99/vi_quad` | Dữ liệu MT dự phòng | Cross-validation |
| Wikipedia Tiếng Việt dump | Mở rộng corpus ngữ cảnh | Crawl + QA filter |
| Zalo AI Challenge datasets | Dữ liệu thực tế thương mại | Fine-tune thêm |
| `bkai-foundation-models/vietnamese-bi-encoder` | Dữ liệu tiếng Việt đa nhiệm | Pre-train context |

### 3.3 Lưu ý kỹ thuật tải

- **Tốc độ:** Dùng `datasets.load_dataset(..., streaming=True)` để kiểm tra nhanh trước khi tải full.
- **Cache:** Dữ liệu sẽ được cache tại `~/.cache/huggingface/datasets/` (~200MB cho full dataset).
- **Format chuyển đổi:** Có thể export sang SQuAD JSON nếu cần dùng với code cũ:
```python
ds = load_dataset("taidng/UIT-ViQuAD2.0")
ds["train"].to_json("viquad2_train.json")
```

---

## 4️⃣ LỘ TRÌNH CẬP NHẬT `dataset_loader.py`

### 4.1 Kiến trúc mới đề xuất

```
dataset_loader.py
├── load_viquad2_0(split="all")       # Tải từ HuggingFace
├── convert_to_squad(json_path)       # Chuyển đổi format nếu cần
├── deduplicate_contexts(ds)          # Loại context trùng train/test
├── validate_answer_spans(ds)         # Kiểm tra answer_start chính xác
├── get_context_split(ds, ratio=...)  # Chia Context-level split
└── load_or_create_viquad(json_path)  # [Giữ lại] hàm cũ để tương thích
```

### 4.2 Chi tiết cập nhật (dự kiến ~100-150 dòng code mới)

**`Hàm 1: `load_viquad2_0()`**
```python
def load_viquad2_0(split="all", cache_dir=None):
    """Tải UIT-ViQuAD 2.0 từ HuggingFace Hub."""
    from datasets import load_dataset
    
    valid_splits = ["train", "validation", "test", "all"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}. Choose from {valid_splits}")
    
    if split == "all":
        # Kết hợp 3 splits (không merge lộn labels)
        return load_dataset("taidng/UIT-ViQuAD2.0")
    else:
        return load_dataset("taidng/UIT-ViQuAD2.0", split=split)
```

**`Hàm 2: `deduplicate_contexts()`**
```python
def deduplicate_contexts(train_ds, test_ds):
    """Loại bỏ context trùng lặp giữa train và test để tránh data leakage."""
    test_contexts = set(test_ds["context"])
    train_filtered = train_ds.filter(
        lambda x: x["context"] not in test_contexts
    )
    return train_filtered, test_ds
```

**`Hàm 3: `validate_answer_spans()`**
```python
def validate_answer_spans(dataset):
    """Kiểm tra answer_start có khớn với text trong context không."""
    invalid = 0
    for example in dataset:
        ctx = example["context"]
        for ans_text, ans_start in zip(
            example["answers"]["text"],
            example["answers"]["answer_start"]
        ):
            if ctx[ans_start:ans_start+len(ans_text)] != ans_text:
                invalid += 1
    print(f"[CHECK] {invalid} invalid answer positions found")
    return dataset
```

**`Hàm 4: `to_squad_format()`** — Chuyển đổi sang format JSON cũ nếu cần tương thích:
```python
def to_squad_format(hf_dataset, output_path):
    """Chuyển đổi từ HuggingFace format sang SQuAD JSON format."""
    # Nhóm theo title -> paragraphs -> qas
    # Output format giống viquad_sample.json
```

### 4.3 Migration plan (bản đồ nâng cấp)

| Bước | Hành động | Thời gian | Trách nhiệm |
| :--- | :--- | :--- | :--- |
| 1 | Chèn hàm `load_viquad2_0()` parallely | 2 ngày | chu trach CS116 |
| 2 | Thêm `deduplicate_contexts()` + `validate_answer_spans()` | 2 ngày | chu trach CS116 |
| 3 | Cập nhật `train_phobert_qa.py` để dùng format mới | 3 ngày | chu trach CS116 + chu trach CS221 |
| 4 | Cập nhật `app_streamlit.py` để load full dataset | 2 ngày | chu trach CS106 (Web UI) |
| 5 | Chạy training với full dataset, so sánh kết quả | 5 ngày | Toàn bộ nhóm |
| 6 | Cập nhật `error_analysis.md` với kết quả mới | 1 ngày | Nhóm |

### 4.4 Các file cần sửa

| File | Thay đổi | Phức tạp |
| :--- | :--- | :--- |
| `dataset_loader.py` | +4 hàm mới, ~100 dòng | ★★☆ |
| `train_phobert_qa.py` | Cập nhật data loading (chuyển đổi format) | ★★★ |
| `eval_metrics.py` | Hỗ trợ is_impossible | ★☆☆ |
| `app_streamlit.py` | Cập nhật UI load dataset | ★★☆ |
| `context.md` | Cập nhật phần dataset sources | ★☆☆ |

---

## 5️⃣ DỰ BÁO THỜI GIAN VÀ NGUỒN LỰC

### 5.1 Dự báo thời gian (tính từ hiện tại)

| Hoạt động | Thời gian dự kiến | Lưu ý |
| :--- | :--- | :--- |
| **Cập nhật `dataset_loader.py`** | 2 ngày | Đang có sẵn API datasets |
| **Cập nhật `train_phobert_qa.py`** | 3 ngày | Cần xử lý format conversion |
| **Chạy thử nghiệm Training** | 3-5 ngày | Phụ thuộc phần cứng (CPU/GPU) |
| **Cập nhật `app_streamlit.py`** | 2 ngày | Giao diện load dataset UI |
| **E2E Testing & Validation** | 2 ngày | Kiểm tra data leakage, EM/F1 |
| **Tổng cộng** | **12-14 ngày** | Có thể rút ngắn song song nếu đủ máy |

### 5.2 Nguồn lực cần thiết

**Phần cứng:**
- **GPU (ưu tiên):** 1 máy có GPU (Tesla T4 hoặc tương đương) để train nhanh (~10x so với CPU).
- **Bộ nhớ RAM:** Tối thiểu 8GB (full dataset ~200MB, training model cần ~4GB).
- **Ổ đĩa:** ~500MB trống (dataset cache + pre-trained model weights ~350MB).

**Phần mềm:**
- Python 3.10+
- `datasets>=2.14`, `transformers>=4.30`, `torch>=2.0`
- `vietnamese-segmenter` (VnCoreNLP hoặc RDRSegmenter) — cho preprocessing
- Streamlit (cho web app)

**Con người:**
| Vai trò | Giờ dự kiến | Nhiệm vụ |
| :--- | :--- | :--- |
| Data Scientist (Tôi - CS116) | 20-25h | Code dataset_loader, convert format |
| NLP Researcher (CS221) | 15-20h | Train pipeline, fine-tune, eval |
| Legal Tech Specialist (CS106) | 10-15h | Streamlit UI, demo integration |
| **Tổng cộng** | **45-60h** | — |

### 5.3 Rủi ro & đề xuất giảm thiểu

| Rủi ro | Xác suất | Ảnh hưởng | Giải pháp |
| :--- | :--- | :--- | :--- |
| Dataset HF bị giới hạn download | Thấp | Hỏng tiến độ train | Download sẵn về local mirror |
| Training time quá lâu (CPU) | Cao nếu không có GPU | Trễ 5-7 ngày | Dùng Colab Pro / AWS spot instance |
| Data leakage ở test set | Đã phát hiện | Sai số liệu eval | Áp dụng `deduplicate_contexts()` |
| Tokenizer lỗi với từ ghép | Trung bình | Giảm EM/F1 | Thêm RDRSegmenter preprocessing |

---

## 📌 KẾ HOẠCH HÀNH Động (Ngay)

1. ✅ **Tải full dataset về local cache** để đảm bảo offline training:
   ```bash
   python3 -c "from datasets import load_dataset; load_dataset('taidng/UIT-ViQuAD2.0')"
   ```
2. 🔧 **Cập nhật `dataset_loader.py`** theo bản đồ nâng cấp ở mục 4.3 (Bước 1-2).
3. 👥 **Báo cho captain & teammates CS221, CS106** để phối hợp cập nhật pipeline.

---

## 📎 PHỤ LỤC: Mẫu code tải và kiểm tra nhanh

```python
from datasets import load_dataset
import json

# 1. Tải dataset
ds = load_dataset("taidng/UIT-ViQuAD2.0")
print("Splits:", list(ds.keys()))
print("Train size:", len(ds["train"]))
print("Val size:", len(ds["validation"]))
print("Test size:", len(ds["test"]))

# 2. Kiểm tra schema
print("Features:", ds["train"].features)

# 3. Xem một ví dụ
example = ds["train"][0]
print(json.dumps(example, indent=2, ensure_ascii=False))

# 4. Kiểm tra data leakage
train_titles = set(ds["train"]["title"])
test_titles = set(ds["test"]["title"])
val_titles = set(ds["validation"]["title"])
print("Train-Test overlap:", len(train_titles & test_titles))
print("Val-Test overlap:", len(val_titles & test_titles))
```
