# -*- coding: utf-8 -*-
"""Streamlit demo: Vietnamese Extractive MRC (CS116 - Do an T11).

Run:  streamlit run app_streamlit.py

Both models expose the same predict(context, question) interface, so the UI is
model-agnostic. The transformer is loaded lazily and cached; the first selection
downloads the checkpoint.
"""
import streamlit as st

from mrc.baseline import TFIDFBaseline

st.set_page_config(page_title="Hệ thống Hỏi-Đáp Tiếng Việt (T11)", page_icon="🇻🇳",
                   layout="wide")

st.title("Hệ thống Đọc hiểu & Trả lời Câu hỏi Tiếng Việt (T11)")
st.caption("CS116 - Lập trình Python cho Máy học | Đề tài T11: Extractive MRC | "
           "Dataset: UIT-ViQuAD 2.0")

st.sidebar.header("Cấu hình")
model_choice = st.sidebar.radio(
    "Chọn mô hình:",
    ("TF-IDF Baseline (nhanh)", "Transformer QA (chính xác hơn)"),
)


@st.cache_resource(show_spinner=True)
def load_model(choice: str):
    if choice.startswith("Transformer"):
        from mrc.qa_model import TransformerQA
        return TransformerQA()
    return TFIDFBaseline()


SAMPLE_CONTEXT = (
    "Trường Đại học Công nghệ Thông tin là một trường đại học thành viên của "
    "Đại học Quốc gia Thành phố Hồ Chí Minh, được thành lập theo Quyết định số "
    "134/2006/QĐ-TTg ngày 8 tháng 6 năm 2006 của Thủ tướng Chính phủ. Trường có "
    "nhiệm vụ đào tạo nguồn nhân lực công nghệ thông tin chất lượng cao."
)
SAMPLE_QUESTION = "Trường Đại học Công nghệ Thông tin được thành lập vào ngày nào?"

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Văn bản ngữ cảnh (Context)")
    context_input = st.text_area("Nhập đoạn văn bản:", value=SAMPLE_CONTEXT, height=220)
    st.subheader("Câu hỏi (Question)")
    question_input = st.text_input("Nhập câu hỏi:", value=SAMPLE_QUESTION)
    submit = st.button("Trích xuất câu trả lời", type="primary")

with col2:
    st.subheader("Kết quả")
    if submit and context_input.strip() and question_input.strip():
        model = load_model(model_choice)
        with st.spinner("Đang xử lý..."):
            answer = model.predict(context_input, question_input)
        if answer:
            st.success(f"**Câu trả lời:** {answer}")
            pos = context_input.find(answer)
            if pos != -1:
                highlighted = (
                    context_input[:pos]
                    + f"<mark style='background:#fde047;padding:2px 4px;border-radius:4px;"
                      f"font-weight:bold'>{answer}</mark>"
                    + context_input[pos + len(answer):]
                )
                st.markdown(f"**Vị trí trong ngữ cảnh:**<br>{highlighted}",
                            unsafe_allow_html=True)
        else:
            st.warning("Mô hình không tìm thấy câu trả lời trong ngữ cảnh.")
        st.caption(f"Mô hình: {model.name}")
    else:
        st.info("Nhập ngữ cảnh và câu hỏi, sau đó nhấn nút để nhận kết quả.")

with st.expander("Về hệ thống"):
    st.markdown(
        "- **Bài toán:** Extractive MRC — trích xuất answer span từ context.\n"
        "- **Baseline:** TF-IDF + cosine similarity (truy hồi câu).\n"
        "- **Transformer:** mô hình QA đa ngữ (XLM-R/SQuAD2) chạy suy luận (inference-only) trên CPU.\n"
        "- **Số liệu thực nghiệm:** xem `results/eval_results.json` — mọi con số đều "
        "được sinh từ lần chạy thực tế trên tập validation, không có số liệu bịa."
    )
