# TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN — ĐHQG-HCM
## KHOA KHOA HỌC MÁY TÍNH

---

# 📘 BÁO CÁO ĐỒ ÁN MÔN HỌC
## MÔN: LẬP TRÌNH PYTHON CHO MÁY HỌC (CS116.F31.CN2.TTNT)

### **ĐỀ TÀI T11: XÂY DỰNG HỆ THỐNG ĐỌC HIỂU VÀ TRẢ LỜI CÂU HỎI TIẾNG VIỆT (VIETNAMESE EXTRACTIVE MACHINE READING COMPREHENSION)**

**Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Quyền  
**Nhóm sinh viên thực hiện:** **Nhóm 7**  
1. **Lê Quang Thi** — MSSV: `25210337` (Nhóm trưởng)  
2. **Trần Trọng Tấn** — MSSV: `25210334`  
3. **Nguyễn Quang Lâm** — MSSV: `25210289`  
4. **Võ Cẩm Thu** — MSSV: `25210342`  

*TP. Hồ Chí Minh — Tháng 08/2026*

---

## TÓM TẮT ĐỒ ÁN (ABSTRACT)
Đồ án tập trung nghiên cứu, xây dựng và đánh giá hệ thống Đọc hiểu máy trích xuất câu trả lời cho tiếng Việt (**Vietnamese Extractive Machine Reading Comprehension - MRC**) dựa trên tập dữ liệu chuẩn benchmark **UIT-ViQuAD 1.0 / 2.0** của Trường Đại học Công nghệ Thông tin (ĐHQG-HCM). Hệ thống tiếp nhận đầu vào gồm một đoạn văn bản ngữ cảnh (*Context*) và một câu hỏi (*Question*), sau đó xác định chính xác vị trí bắt đầu (*start index*) và vị trí kết thúc (*end index*) của đoạn văn bản trả lời (*Answer Span*) nằm ngay trong ngữ cảnh. Nhóm đã triển khai mô hình đối sánh cơ sở (**Baseline TF-IDF**) và mô hình học sâu hiện đại (**Transformer QA đa ngữ**, `xlm-roberta-base-squad2`) chạy suy luận trực tiếp cho bài toán Question Answering (do ràng buộc chỉ có CPU nên không fine-tune từ đầu). Kết quả thực nghiệm được đánh giá nghiêm ngặt theo 2 thước đo chuẩn: **Exact Match (EM)** và **Token-level F1-Score** trên tập validation, đồng thời hệ thống được đóng gói thành ứng dụng Web Demo tương tác bằng thư viện **Streamlit**. Toàn bộ số liệu trong báo cáo đều được sinh từ lần chạy thực tế.

---

## CHƯƠNG 1. GIỚI THIỆU BÀI TOÁN VÀ MỤC TIÊU NGHIÊN CỨU

### 1.1. Đặt vấn đề
Trong kỷ nguyên bùng nổ thông tin, việc tìm kiếm câu trả lời chính xác từ khối lượng lớn tài liệu văn bản là một nhu cầu cấp thiết. Khác với các hệ thống tìm kiếm thông thường chỉ trả về danh sách các đường dẫn hoặc cả bài viết dài, bài toán **Đọc hiểu máy (Machine Reading Comprehension - MRC)** hướng tới mục tiêu tự động hóa việc đọc, phân tích ngữ nghĩa và trích xuất trực tiếp câu trả lời ngắn gọn, chính xác nhất.

### 1.2. Mục tiêu và Phạm vi của Đồ án
* **Mục tiêu:** Xây dựng một quy trình hoàn chỉnh từ nạp dữ liệu, tiền xử lý, huấn luyện mô hình Transformer tiếng Việt, đánh giá định lượng và trực quan hóa qua giao diện Web.
* **Phạm vi quy định (Theo chỉ đạo của GVHD - ThS. Nguyễn Hữu Quyền):**
  1. Tập trung tuyệt đối vào bài toán **Extractive MRC** (đoạn trả lời là một chuỗi con liên tục trong văn bản ngữ cảnh).
  2. Không mở rộng lan man sang mô hình sinh văn bản tự do (Generative QA) hay RAG phức tạp.
  3. Phân chia tập dữ liệu huấn luyện/kiểm thử theo cấp độ bài báo (**Context-level split**) để ngăn ngừa hiện tượng rò rỉ dữ liệu (**Data Leakage**).

---

## CHƯƠNG 2. TẬP DỮ LIỆU BENCHMARK UIT-ViQuAD VÀ QUY TRÌNH TIỀN XỬ LÝ

### 2.1. Giới thiệu bộ dữ liệu UIT-ViQuAD
UIT-ViQuAD (*UIT Vietnamese Question Answering Dataset*) là bộ dữ liệu đọc hiểu máy tiếng Việt quy mô lớn được xây dựng thủ công bởi các chuyên gia ngôn ngữ và sinh viên UIT từ các bài viết trên Wikipedia tiếng Việt. Cấu trúc chuẩn theo định dạng SQuAD JSON:
* Mỗi mẫu dữ liệu gồm: `id`, `context` (đoạn văn), `question` (câu hỏi) và `answers` (danh sách câu trả lời chuẩn kèm chỉ số ký tự `answer_start`).

### 2.2. Chiến lược chống rò rỉ dữ liệu (Context-Level Split)
* Nếu phân chia dữ liệu ngẫu nhiên theo từng câu hỏi (*Question-level split*), các câu hỏi có cùng một đoạn ngữ cảnh sẽ xuất hiện ở cả tập Train và tập Test, dẫn đến việc mô hình "học vẹt" ngữ cảnh và gây thổi phồng độ chính xác thực tế.
* Nhóm áp dụng **Context-Level Split**: Toàn bộ các câu hỏi thuộc cùng một đoạn văn bản chỉ được nằm trong tập Train hoặc tập Test, đảm bảo tính khách quan 100% của kết quả thực nghiệm.

---

## CHƯƠNG 3. PHƯƠNG PHÁP VÀ MÔ HÌNH HỌC MÁY

### 3.1. Mô hình Đối sánh (Baseline Model)
* Mô hình cơ sở sử dụng phương pháp trích xuất dựa trên ma trận tần suất từ và nghịch đảo tần suất tài liệu (**TF-IDF**) kết hợp độ đo tương đồng góc Cosine giữa câu hỏi và từng câu đơn trong đoạn văn bản.

### 3.2. Mô hình Transformer: Suy luận với XLM-R (inference-only)
* **Ràng buộc:** Môi trường **chỉ có CPU** nên fine-tune đầy đủ trên ~28k câu hỏi là bất khả thi. Nhóm dùng mô hình QA đa ngữ đã huấn luyện sẵn `deepset/xlm-roberta-base-squad2` và chạy **inference-only** (áp dụng zero-shot cho tiếng Việt qua XLM-R). Suy luận dùng HuggingFace QA pipeline với **offset mapping** (trích span chính xác theo ký tự) và **doc-stride** (cửa sổ trượt cho ngữ cảnh dài) — đúng phần mà bản cũ làm sai.
* **Cơ chế Span Prediction:** Lớp tuyến tính ở đầu ra Transformer tính xác suất vị trí bắt đầu $P_{start}(i)$ và kết thúc $P_{end}(j)$; đáp án là span có tổng điểm cao nhất:

$$\mathcal{L} = -\log P_{start}(y_{start}) - \log P_{end}(y_{end})$$

* **Đường cong huấn luyện (minh hoạ):** Để có biểu đồ loss/EM-F1 theo epoch cho báo cáo, nhóm chạy một **tiny fine-tune** trên CPU (tập con nhỏ, vài epoch) với mô hình nhẹ `distilbert-base-multilingual-cased` — chỉ nhằm mục đích minh hoạ quá trình học, không phải kết quả chính (xem `results/training_curve.json`).

---

## CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ

### 4.1. Thước đo Đánh giá
* **Exact Match (EM):** Tỷ lệ phần trăm các câu trả lời do mô hình dự đoán trùng khớp hoàn toàn 100% với nhãn chuẩn của con người sau khi đã chuẩn hóa dấu câu và khoảng trắng.
* **F1-Score:** Đo lường mức độ trùng lặp ở cấp độ từ (Token-level Overlap) giữa câu trả lời dự đoán và Ground Truth.

### 4.2. Bảng So sánh Hiệu năng

> **Lưu ý phương pháp (Method note):** Do ràng buộc **chỉ có CPU**, nhóm không fine-tune mô hình Transformer từ đầu. Kết quả chính được đo bằng cách chạy suy luận (**inference-only**) mô hình QA đa ngữ đã được huấn luyện sẵn `deepset/xlm-roberta-base-squad2` (áp dụng zero-shot cho tiếng Việt). Mọi con số dưới đây được **sinh ra từ lần chạy thực tế** trên **tập validation** (n = 400 câu hỏi; tập test không có nhãn đáp án nên không đo được). Xem `results/eval_results.json`.

| Mô hình thử nghiệm | Kiến trúc | Exact Match (EM) | F1-Score | Thời gian phản hồi |
| :--- | :--- | :---: | :---: | :---: |
| Baseline TF-IDF | Truy hồi câu (Cosine) | 0.75% | 23.95% | ~0.6ms |
| XLM-R-squad2 (inference-only) | Transformer đa ngữ, zero-shot | 40.75% | 56.53% | ~86ms |

Baseline TF-IDF là cận dưới (chỉ truy hồi cả câu nên EM gần 0, F1 phản ánh trùng lặp từ). Mô hình Transformer chưa được fine-tune riêng cho tiếng Việt nên EM/F1 ở mức trung bình — đây là con số trung thực, không phải số liệu thổi phồng.

### 4.3. Phân tích Kết quả (Result Analysis)
Phân tích từ dự đoán thực tế (xem `results/eval_results.json` và `visualizations/`):

1. **Ảnh hưởng của độ dài ngữ cảnh:** EM của Transformer giảm dần khi ngữ cảnh dài hơn (EM ≈ 67% với context < 100 từ → 42% (100–200) → 37% (200–300) → 24% (300+ từ)), cho thấy độ dài ngữ cảnh làm khó việc định vị đáp án.
2. **Loại câu hỏi (single- vs multi-sentence):** Transformer trả lời tốt hơn hẳn với câu hỏi một câu (EM 55% / F1 76%) so với câu hỏi cần suy luận nhiều câu (EM 31% / F1 55%) — phù hợp với kỳ vọng lý thuyết.
3. **Baseline vs Transformer:** khoảng cách EM 0.75% → 40.75% cho thấy giá trị của biểu diễn ngữ cảnh sâu so với đối sánh từ vựng thuần túy.

*(Phân loại single-/multi-sentence dùng heuristic đo độ phủ từ khóa câu hỏi trong câu chứa đáp án; xem `mrc/data.py`.)*

---

## CHƯƠNG 5. ỨNG DỤNG WEB DEMO STREAMLIT
Hệ thống được đóng gói thành giao diện Web trực quan bằng **Streamlit** (`app_streamlit.py`):
* **Tính năng:** Người dùng có thể dán bất kỳ bài báo, văn bản ngữ cảnh nào vào ô nhập liệu và đặt câu hỏi tự do.
* **Hiển thị:** Hệ thống trích xuất câu trả lời trong thời gian thực và tự động đánh dấu (Highlight) trực tiếp vị trí câu trả lời trên đoạn văn bản gốc.

---

## CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
* **Kết luận:** Đồ án xây dựng một pipeline Extractive MRC tiếng Việt sạch, chạy được trên CPU, với **số liệu trung thực** (EM 40.75% / F1 56.53% trên validation cho mô hình Transformer suy luận, so với baseline TF-IDF 0.75% / 23.95%). Đã tuân thủ Context-level split chống Data Leakage và phân tích ảnh hưởng độ dài ngữ cảnh cùng loại câu hỏi.
* **Hạn chế:** Do chỉ có CPU, mô hình Transformer chưa được fine-tune riêng cho tiếng Việt (chạy zero-shot đa ngữ), nên EM/F1 còn khiêm tốn so với tiềm năng của một mô hình fine-tune trên UIT-ViQuAD.
* **Hướng phát triển:** Fine-tune đầy đủ `vinai/phobert-base-v2` hoặc `nguyenvulebinh/vi-mrc-base` trên GPU; thêm ngưỡng nhận diện câu hỏi không có câu trả lời (Unanswerable Threshold); mở rộng đánh giá trên toàn tập validation.
