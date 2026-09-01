# CS116.F31.CN2.TTNT - Vietnamese MRC (Machine Reading Comprehension)

## Overview
Đồ án môn học CS116 - Lập trình Python cho Học máy nâng cao  
**Đề tài T11**: Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt

## Team
- **Trần Trọng Tấn** - 25210334  
- **Nguyễn Quang Lâm** - 25210289  
- **Giảng viên hướng dẫn**: ThS. Nguyễn Hữu Quyền

## Features
✅ Pipeline huấn luyện PhoBERT trên UIT-ViQuAD 2.0 (39,569 QA pairs)  
✅ So sánh hiệu năng: TF-IDF Baseline vs PhoBERT-base vs ViDeBERTa  
✅ Đánh giá bằng Exact Match (EM) và F1-Score  
✅ Deep Error Analysis với phân loại chi tiết lỗi  
✅ Web demo tương tác bằng Streamlit  

## Dataset
- **UIT-ViQuAD 2.0**: Bộ dữ liệu QA tiếng Việt chuẩn (28,454 train / 3,814 val / 7,301 test)
- **Nguồn thay thế**: HuggingFace, Kaggle, Wikipedia tiếng Việt

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Huấn luyện mô hình
python train_phobert_qa.py

# Chạy demo web
streamlit run app_streamlit.py

# Đánh giá mô hình
python -c "from eval_metrics import evaluate_predictions; print(evaluate_predictions(...))"
```

## Project Structure
├── dataset_loader.py    # Tải và tiền xử lý UIT-ViQuAD 2.0
├── train_phobert_qa.py  # Pipeline huấn luyện PhoBERT
├── baseline_model.py    # Mô hình TF-IDF baseline
├── eval_metrics.py      # Đánh giá EM/F1
├── app_streamlit.py     # Giao diện web demo
├── viquad_sample.json   # Dữ liệu mẫu
└── error_analysis.md    # Phân tích lỗi mô hình

## Results
| Model | Exact Match | F1-Score |
|-------|-------------|----------|
| TF-IDF Baseline | ~25% | ~35% |
| PhoBERT-base-v2 | ~62% | ~74% |
| ViDeBERTa-base | ~65% | ~77% |

## License
Academic use only @ UIT
