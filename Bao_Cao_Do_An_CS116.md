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
Đồ án tập trung nghiên cứu, xây dựng và đánh giá hệ thống Đọc hiểu máy trích xuất câu trả lời cho tiếng Việt (**Vietnamese Extractive Machine Reading Comprehension - MRC**) dựa trên tập dữ liệu chuẩn benchmark **UIT-ViQuAD 1.0 / 2.0** của Trường Đại học Công nghệ Thông tin (ĐHQG-HCM). Hệ thống tiếp nhận đầu vào gồm một đoạn văn bản ngữ cảnh (*Context*) và một câu hỏi (*Question*), sau đó xác định chính xác vị trí bắt đầu (*start index*) và vị trí kết thúc (*end index*) của đoạn văn bản trả lời (*Answer Span*) nằm ngay trong ngữ cảnh. Nhóm đã triển khai mô hình đối sánh cơ sở (**Baseline TF-IDF**) và mô hình học sâu hiện đại (**PhoBERT-base-v2**) được fine-tune chuyên biệt cho bài toán Question Answering. Kết quả thực nghiệm được đánh giá nghiêm ngặt theo 2 thước đo chuẩn: **Exact Match (EM)** và **Token-level F1-Score**, đồng thời hệ thống được đóng gói thành ứng dụng Web Demo tương tác bằng thư viện **Streamlit**.

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

### 3.2. Mô hình Transformer: Fine-tuning PhoBERT-base-v2
* **Kiến trúc PhoBERT:** Mô hình ngôn ngữ tiền huấn luyện theo kiến trúc RoBERTa được đào tạo trên 20GB văn bản tiếng Việt chuẩn từ VinAI Research.
* **Cơ chế Span Prediction:** Thêm một lớp tuyến tính (Linear Classification Head) ở đầu ra của Transformer để tính xác suất cho vị trí bắt đầu $P_{start}(i)$ và vị trí kết thúc $P_{end}(j)$ của câu trả lời:

$$\mathcal{L} = -\log P_{start}(y_{start}) - \log P_{end}(y_{end})$$

---

## CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ

### 4.1. Thước đo Đánh giá
* **Exact Match (EM):** Tỷ lệ phần trăm các câu trả lời do mô hình dự đoán trùng khớp hoàn toàn 100% với nhãn chuẩn của con người sau khi đã chuẩn hóa dấu câu và khoảng trắng.
* **F1-Score:** Đo lường mức độ trùng lặp ở cấp độ từ (Token-level Overlap) giữa câu trả lời dự đoán và Ground Truth.

### 4.2. Bảng So sánh Hiệu năng

| Mô hình thử nghiệm | Kiến trúc | Exact Match (EM) | F1-Score | Thời gian phản hồi |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline TF-IDF** | Heuristic Lexical Matching | 28.4% | 46.2% | < 5ms |
| **PhoBERT-base-v2 (QA)** | **Pre-trained Transformer Fine-tuned** | **68.7%** | **84.5%** | ~45ms |

### 4.3. Phân tích Lỗi sai (Error Analysis)
Nhóm đã trích xuất các mẫu dự đoán sai và phân loại thành các nhóm nguyên nhân:
1. **Lệch biên từ đa âm tiết (38%):** Cơ chế tách từ BPE cắt ngang từ ghép tiếng Việt (ví dụ: mất tiền tố chức danh "GS.TS").
2. **Trích xuất thừa/thiếu ngữ cảnh (27%):** Mô hình lấy dư trạng từ hoặc đại từ chỉ định.
3. **Nhầm lẫn thực thể cùng loại (18%):** Nhầm lẫn giữa các mốc thời gian hoặc địa danh gần nhau trong đoạn văn.

---

## CHƯƠNG 5. ỨNG DỤNG WEB DEMO STREAMLIT
Hệ thống được đóng gói thành giao diện Web trực quan bằng **Streamlit** (`app_streamlit.py`):
* **Tính năng:** Người dùng có thể dán bất kỳ bài báo, văn bản ngữ cảnh nào vào ô nhập liệu và đặt câu hỏi tự do.
* **Hiển thị:** Hệ thống trích xuất câu trả lời trong thời gian thực và tự động đánh dấu (Highlight) trực tiếp vị trí câu trả lời trên đoạn văn bản gốc.

---

## CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
* **Kết luận:** Đồ án đã hoàn thành xuất sắc các mục tiêu đề ra, tuân thủ nghiêm ngặt định hướng của ThS. Nguyễn Hữu Quyền về bài toán Extractive MRC và phòng chống Data Leakage.
* **Hướng phát triển:** Tích hợp mô hình ViDeBERTa-v3 và kết hợp bộ nhận diện câu hỏi không có câu trả lời (Unanswerable Verification Threshold).
