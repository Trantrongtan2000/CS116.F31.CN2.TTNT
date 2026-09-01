# 📋 CONTEXT & HƯỚNG DẪN DỰ ÁN CHO AI AGENTS — CS116

> **Môn học:** CS116 — Lập Trình Python Cho Máy Học Nâng Cao (Advanced Machine Learning)  
> **Nhóm thực hiện:** Nhóm 7 (Trần Trọng Tấn - 25210334, Nguyễn Quang Lâm - 25210289)  
> **Giảng viên phụ trách:** ThS. Nguyễn Hữu Quyền — Trường Đại học Công nghệ Thông tin (ĐHQG-HCM)

---

## 🎯 1. NỘI DUNG ĐỒ ÁN
* **Tên đề tài chính thức:** **Đề tài T11 — Đọc hiểu máy trích xuất tiếng Việt (Vietnamese Extractive Machine Reading Comprehension - MRC)**
* **Mục tiêu kỹ thuật:** 
  * Xây dựng pipeline hoàn chỉnh từ tiền xử lý, huấn luyện đến đánh giá mô hình hỏi đáp trích xuất.
  * Nhận đầu vào là đoạn văn bản ngữ cảnh ($Context$) và câu hỏi ($Question$), dự đoán vị trí $(start\_idx, end\_idx)$ của câu trả lời ($Answer$) nằm trong đoạn văn.
  * So sánh hiệu năng mô hình cơ sở (Baseline TF-IDF / BM25) với các mô hình tiền huấn luyện sâu: `vinai/phobert-base-v2`, `FPTAI/videberta-base`.
  * Đo lường hiệu năng bằng **Exact Match (EM)** và **F1-Score**.
  * Đóng gói Web App tương tác Demo bằng **Streamlit**.

---

## 📊 2. BỘ DỮ LIỆU (DATASET) & NGUỒN TÌM DATASET

### Bộ dữ liệu sử dụng chính:
* **`UIT-ViQuAD 2.0` (Vietnamese Question Answering Dataset)**:
  * Do nhóm nghiên cứu UIT công bố (chuẩn benchmark cho bài toán QA tiếng Việt).
  * Định dạng JSON theo cấu trúc SQuAD: gồm các trường `context`, `qas` (`question`, `id`, `answers` chứa `text`, `answer_start`).
  * **Quy tắc quan trọng:** Chia tập Train/Dev/Test theo **Context-level** (các câu hỏi cùng 1 đoạn văn phải cùng thuộc 1 tập) để tránh rò rỉ dữ liệu (*Data Leakage*).

### Nguồn tìm kiếm & tải dataset mở rộng (Dataset Sources):
1. **Hugging Face Hub**:
   * [`uitnlp/vietnamese_students_feedback`](https://huggingface.co/datasets)
   * `bkai-foundation-models/vietnamese-bi-encoder`
   * Tìm kiếm keyword: `vietnamese-mrc`, `viquad`, `vietnamese-qa` trên [Hugging Face Datasets](https://huggingface.co/datasets?search=viquad).
2. **Kaggle Datasets**:
   * Bộ dữ liệu Zalo AI Challenge (E-Commerce QA, Elementary QA).
   * UIT Data Challenge / ViQuAD Benchmark trên Kaggle.
3. **UIT NLP / Data Science Lab Repositories**:
   * Trang công bố nghiên cứu của nhóm UIT NLP: [UIT-ViQuAD](https://github.com/VinAIResearch/PhoBERT) & [NLP@UIT](https://github.com/uitnlp).
4. **Văn bản pháp luật / Wikipedia Tiếng Việt (Crawler)**:
   * Wikipedia Dump tiếng Việt cho việc mở rộng corpus ngữ cảnh.

---

## 👨‍🏫 3. TÍNH CÁCH & TIÊU CHÍ ĐÁNH GIÁ CỦA GIẢNG VIÊN (ThS. NGUYỄN HỮU QUYỀN)

* **Phong cách giảng dạy:** Thực dụng, hướng đến tính ứng dụng thực tế (*Industry-ready*), chú trọng sản phẩm hoàn thiện và quy trình chuẩn chỉ (*MLOps & Clean Code*).
* **Gu đánh giá & Tiêu chí chấm điểm:**
  1. **Sản phẩm chạy được thực tế (Live Demo):** Rất thích Web App Demo có giao diện trực quan (Streamlit/Gradio), cho phép nhập thử đoạn văn và câu hỏi bất kỳ để test trực tiếp.
  2. **Trực quan hóa số liệu:** Báo cáo phải có biểu đồ đường biểu diễn quá trình học (Loss curve, Accuracy/EM/F1 curve qua từng Epoch), ma trận phân tích kết quả.
  3. **Cấu trúc báo cáo:** Trình bày sạch sẽ, khoa học, đầy đủ phần Giới thiệu, Cơ sở lý thuyết, Phương pháp đề xuất, Thực nghiệm và Đánh giá kết quả (Báo cáo Word/PDF 30–50 trang).
  4. **Codebase sạch sẽ:** Mã nguồn viết module rõ ràng (`dataset_loader.py`, `train.py`, `eval.py`, `app.py`), có file `requirements.txt` và `README.md` hướng dẫn chạy.

---

## 📂 4. DANH MỤC FILE & VAI TRÒ
* [`dataset_loader.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/00_DO_AN_TONG_HOP/01_CS116_DoAn_T11_MRC/dataset_loader.py): Đọc và xử lý UIT-ViQuAD.
* [`baseline_model.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/00_DO_AN_TONG_HOP/01_CS116_DoAn_T11_MRC/baseline_model.py): Mô hình cơ sở so khớp TF-IDF / BM25.
* [`train_phobert_qa.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/00_DO_AN_TONG_HOP/01_CS116_DoAn_T11_MRC/train_phobert_qa.py): Pipeline huấn luyện mô hình Transformer.
* [`eval_metrics.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/00_DO_AN_TONG_HOP/01_CS116_DoAn_T11_MRC/eval_metrics.py): Đánh giá EM và F1.
* [`app_streamlit.py`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/00_DO_AN_TONG_HOP/01_CS116_DoAn_T11_MRC/app_streamlit.py): Giao diện tương tác trực tiếp.
* [`Bao_Cao_Do_An_CS116.docx`](file:///home/tan/04_Studies_Knowledge/UIT_Studies/Các môn học kỳ 3/00_DO_AN_TONG_HOP/01_CS116_DoAn_T11_MRC/Bao_Cao_Do_An_CS116.docx): Báo cáo kỹ thuật nộp kết thúc môn.
