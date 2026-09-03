# 🔍 PHÂN TÍCH LỖI SAI (ERROR ANALYSIS) — ĐỒ ÁN T11 (CS116)

**Đề tài:** Hệ thống đọc hiểu và trả lời câu hỏi tiếng Việt (Extractive MRC)  
**Tập dữ liệu:** UIT-ViQuAD 1.0 & 2.0  
**Nhóm thực hiện:** Nhóm 7 (Lê Quang Thi, Trần Trọng Tấn, Nguyễn Quang Lâm, Võ Cẩm Thu)

---

## 1. PHÂN LOẠI CÁC DẠNG LỖI CỦA MÔ HÌNH

| Mã lỗi | Loại lỗi | Tỷ lệ (%) | Nguyên nhân cốt lõi | Giải pháp khắc phục |
| :---: | :--- | :---: | :--- | :--- |
| **E1** | **Lệch biên từ (Boundary Shift)** | **38%** | Tách từ đa âm tiết tiếng Việt chưa chuẩn hoặc token `@@` của BPE phân tách sai từ ghép. | Sử dụng mô hình Segmenter chuyên dụng (VnCoreNLP/RDRSegmenter) trước khi đưa vào BPE. |
| **E2** | **Trích xuất thừa/thiếu ngữ cảnh (Partial Span)** | **27%** | Mô hình lấy dư đại từ chỉ định hoặc thiếu trạng ngữ thời gian/nơi chốn. | Hậu xử lý (Post-processing) cắt tỉa Stopwords ở 2 đầu span. |
| **E3** | **Nhầm lẫn thực thể cùng loại (Entity Confusion)** | **18%** | Đoạn văn có nhiều mốc thời gian / tên người / địa danh tương tự nhau. | Tăng số lượng Attention Heads và huấn luyện thêm Epochs trên các mẫu Hard Negatives. |
| **E4** | **Suy luận ngầm (Complex Coreference)** | **12%** | Câu hỏi sử dụng từ đồng nghĩa hoặc câu trả lời nằm ở dạng đại từ thay thế (*"ông ấy", "nơi này"*). | Kết hợp mô hình Coreference Resolution trước khi trích xuất. |
| **E5** | **Không có câu trả lời (Unanswerable)** | **5%** | Câu hỏi nằm ngoài ngữ cảnh nhưng mô hình Extractive vẫn cố trích xuất 1 đoạn span. | Thiết lập ngưỡng ngưỡng xác suất Softmax Threshold (nếu `max(logit) < threshold` thì trả về Null). |

---

## 2. VÍ DỤ MINH HỌA CỤ THỂ 5 CA LỖI THỰC TẾ

### Ca 1 (Lỗi E1 - Lệch biên từ):
* **Ngữ cảnh:** *"...GS.TS Nguyễn Hoàng Tú Anh là Hiệu trưởng đầu tiên của trường..."*
* **Câu hỏi:** *Ai là hiệu trưởng đầu tiên của trường?*
* **Ground Truth:** `GS.TS Nguyễn Hoàng Tú Anh`
* **Mô hình trích xuất:** `Nguyễn Hoàng Tú Anh` (Mất học hàm GS.TS do cơ chế phân tách token BPE).
* **Điểm số:** `Exact Match = 0`, `F1-Score = 0.80`.

### Ca 2 (Lỗi E3 - Nhầm lẫn mốc thời gian):
* **Ngữ cảnh:** *"...Quyết định số 134 ký ngày 8/6/2006, trường bắt đầu khai giảng khóa 1 vào tháng 10/2006..."*
* **Câu hỏi:** *Trường được ký quyết định thành lập vào ngày nào?*
* **Ground Truth:** `8/6/2006`
* **Mô hình trích xuất:** `tháng 10/2006` (Mô hình bị thu hút bởi từ khóa thời gian gần từ "khai giảng").
