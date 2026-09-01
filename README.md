# CS116.F31.CN2.TTNT — Vietnamese Extractive Machine Reading Comprehension (MRC)

## 📌 Tên đề tài chính thức

**Đề tài T11 — Đọc hiểu máy trích xuất tiếng Việt (Vietnamese Extractive Machine Reading Comprehension - MRC)**

> Đồ án môn học **CS116 — Lập trình Python cho Máy học Nâng Cao (Advanced Machine Learning)**

## 👥 Thành viên nhóm

| Họ và tên | MSSV | Vai trò |
|-----------|------|---------|
| **Trần Trọng Tấn** | 25210334 | Thành viên |
| **Nguyễn Quang Lâm** | 25210289 | Thành viên |

- **Giảng viên hướng dẫn**: ThS. Nguyễn Hữu Quyền
- **Trường**: Đại học Công nghệ Thông tin — ĐHQG-HCM

---

## 🎯 Mục tiêu dự án

Xây dựng pipeline hoàn chỉnh từ tiền xử lý, huấn luyện đến đánh giá mô hình hỏi đáp trích xuất cho tiếng Việt. Hệ thống tiếp nhận đầu vào gồm đoạn văn bản ngữ cảnh (*Context*) và câu hỏi (*Question*), sau đó dự đoán vị trí bắt đầu (*start_idx*) và kết thúc (*end_idx*) của câu trả lời nằm trong ngữ cảnh.

## 📊 Bộ dữ liệu

### UIT-ViQuAD 2.0
- **Nguồn**: [`taidng/UIT-ViQuAD2.0`](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) (HuggingFace Hub)
- **Định dạng**: SQuAD 2.0 (có `is_impossible`, `plausible_answers`)
- **Quy mô**: 39,569 cặp câu hỏi - câu trả lời trên 6,399 đoạn văn

| Split | QA pairs | Contexts | Titles | Impossible QA |
|-------|----------|----------|--------|---------------|
| Train | 28,454 | 4,101 | 138 | 9,216 (32.38%) |
| Validation | 3,814 | 557 | 19 | 1,161 (30.44%) |
| Test | 7,301 | 1,241 | 48 | 0 (0.00%) |
| **Total** | **39,569** | **6,399** | **197** | **10,377** |

- **Chiến lược chia dữ liệu**: Context-level split — các câu hỏi cùng một đoạn văn chỉ thuộc một tập duy nhất để tránh rò rỉ dữ liệu (*Data Leakage*).

## 🤖 Mô hình & Công nghệ

### Kiến trúc thử nghiệm

| Mô hình | Kiến trúc | Nguồn |
|---------|-----------|-------|
| **Baseline TF-IDF** | Lexical Matching + Cosine Similarity | scikit-learn |
| **PhoBERT-base-v2** | Pre-trained RoBERTa Fine-tuned | VinAI Research |
| **ViDeBERTa-base** | Pre-trained DeBERTa Fine-tuned | FPT AI |

### Cấu trúc mô hình
- **Span Prediction Head**: Lớp tuyến tính tính xác suất vị trí bắt đầu $P_{start}(i)$ và kết thúc $P_{end}(j)$:

$$\mathcal{L} = -\log P_{start}(y_{start}) - \log P_{end}(y_{end})$$

### Công nghệ sử dụng
- **Framework**: PyTorch, HuggingFace Transformers, Datasets, Evaluate
- **Demo**: Streamlit Web App
- **Tokenization**: PhoBERT tokenizer (vocab size 64,000)
- **Tiền xử lý**: SQuAD format, context-level deduplication

---

## 📈 Kết quả thực nghiệm

### Bảng so sánh hiệu năng

| Mô hình | Exact Match (EM) | F1-Score | Thời gian phản hồi |
|---------|------------------|----------|---------------------|
| TF-IDF Baseline | 28.4% | 46.2% | < 5ms |
| **PhoBERT-base-v2 (QA)** | **68.7%** | **84.5%** | ~45ms |

### Phân tích lỗi sai (Error Analysis)

| Nhóm lỗi | Tỷ lệ | Mô tả |
|----------|-------|-------|
| Lệch biên từ đa âm tiết | 38% | Cơ chế BPE cắt ngang tứ ghép tiếng Việt |
| Trích xuất thừa/thiếu ngữ cảnh | 27% | Lấy dư trạng từ hoặc đại từ chỉ định |
| Nhầm lẫn thực thể cùng loại | 18% | Nhầm giữa mốc thời gian hoặc địa danh |

---

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- RAM ≥ 8GB (khuyến nghị 16GB cho training)
- GPU NVIDIA CUDA hoặc AMD ROCm (tùy chọn, hỗ trợ CPU fallback)

```bash
# Clone repository
git clone <repo-url>
cd 01_CS116_DoAn_T11_MRC

# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA
# HOẶC
pip install torch torchvision torchaudio  # CPU only

pip install transformers datasets accelerate evaluate
pip install streamlit scikit-learn pandas numpy pyvi
```

### ⚠️ Lưu ý về GPU AMD ROCm

Mô hình huấn luyện (PhoBERT-base-v2, 134M tham số) chạy trên **HuggingFace Transformers + PyTorch**:

1. **PyTorch hỗ trợ ROCm**: PyTorch chính thức hỗ trợ AMD ROCm qua build `rocm`:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
   ```

2. **Hạn chế**: Một số kernel attention tối ưu cho Flash Attention 2 / SDPA chỉ được tối ưu cho NVIDIA CUDA. Trên ROCm, fallback về attention chuẩn sẽ chậm hơn ~20-40%.

3. **Khuyến nghị**: Nếu không có GPU, script tự động fallback sang CPU với subset 5,000 mẫu train + 500 val/test (xem `training_output.log`).

4. **Verify GPU**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"ROCm available: {torch.version.hip is not None}")
   ```

---

## 🎮 Sử dụng

### Huấn luyện mô hình
```bash
# Huấn luyện PhoBERT với dataset nhỏ (CPU)
python train_phobert_qa.py

# Huấn luyện với GPU và toàn bộ dataset
python train_phobert_full.py
```

### Chạy Web Demo
```bash
streamlit run app_streamlit.py
```
> Mở trình duyệt tại `http://localhost:8501`, dán đoạn văn và đặt câu hỏi tự do.

### Đánh giá mô hình
```bash
python -c "from eval_metrics import evaluate_predictions; print(evaluate_predictions(...))"
```

### Tải dataset
```python
from dataset_loader import load_viquad2_0

train_data = load_viquad2_0(split="train")      # 28,454 QA
val_data   = load_viquad2_0(split="validation") # 3,814 QA
test_data  = load_viquad2_0(split="test")       # 7,301 QA
```

---

## 📂 Cấu trúc dự án

```
01_CS116_DoAn_T11_MRC/
├── dataset_loader.py         # Tải và tiền xử lý UIT-ViQuAD 2.0
├── train_phobert_qa.py       # Pipeline huấn luyện PhoBERT
├── train_phobert_full.py     # Training với toàn bộ dataset
├── baseline_model.py         # Mô hình TF-IDF baseline
├── eval_metrics.py           # Đánh giá EM/F1
├── app_streamlit.py          # Giao diện web demo
├── viquad_sample.json        # Dữ liệu mẫu (4 QA)
├── viquad2_deduped_*.json    # Dataset deduped (train/val/test)
├── training_output.log       # Log huấn luyện
├── Bao_Cao_Do_An_CS116.*     # Báo cáo chi tiết (.docx/.pdf/.md)
├── error_analysis.md         # Phân tích lỗi
├── visualizations/           # Biểu đồ trực quan hóa
└── models/                   # Checkpoint mô hình
```

---

## 📚 Nguồn tham khảo

- [UIT-ViQuAD 2.0 — HuggingFace](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)
- [PhoBERT — VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [SQuAD 2.0 Paper](https://arxiv.org/abs/1806.03822)

---

## 📄 License

Academic use only @ UIT
