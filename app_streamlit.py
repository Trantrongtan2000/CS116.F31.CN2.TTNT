# -*- coding: utf-8 -*-
"""
Streamlit Web Demo: Vietnamese Extractive QA (Do an T11 - CS116 UIT)
Run command: streamlit run app_streamlit.py
"""
import streamlit as st
from baseline_model import BaselineTFIDFQA
from train_phobert_qa import VietnameseQAModel
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="Hệ thống Hỏi-Đáp Tiếng Việt (T11)", page_icon="🤖", layout="wide")

st.title("🤖 HỆ THỐNG ĐỌC HIỂU & TRẢ LỜI CÂU HỎI TIẾNG VIỆT (T11)")
st.caption("Đồ án môn học: CS116 - Lập trình Python cho Máy học (GVHD: ThS. Nguyễn Hữu Quyền) | Nhóm 7")

# Sidebar for model selection
st.sidebar.header("⚙️ Cấu hình hệ thống")
model_choice = st.sidebar.radio(
    "Chọn mô hình:",
    ("PhoBERT Transformer (Full)', 'TF-IDF Baseline (Fast)'),
    help="Chọn mô hình để dự đoán câu trả lời"
)

# Cache model loading
@st.cache_resource
def load_model(model_type):
    if model_type == "PhoBERT Transformer (Full)":
        return VietnameseQAModel()
    else:
        return BaselineTFIDFQA()

qa_engine = load_model(model_choice)

# Main interface
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📄 Văn bản Ngữ cảnh (Context)")
    sample_context = (
        "Trường Đại học Công nghệ Thông tin là một trường đại học thành viên của Đại học Quốc gia "
        "Thành phố Hồ Chí Minh, được thành lập theo Quyết định số 134/2006/QĐ-TTg ngày 8 tháng 6 năm 2006 "
        "của Thủ tướng Chính phát. Trường có nhiệm vụ đào tạo nguồn nhân lực công nghệ thông tin chất lượng cao, "
        "đóng góp tích cực vào sự phát triển của nền công nghiệp công nghệ thông tin Việt Nam."
    )
    context_input = st.text_area("Nhập đoạn văn bản:", value=sample_context, height=200)
    
    st.subheader("❓ Câu hỏi (Question)")
    sample_question = "Trường Đại học Công nghệ Thông tin được thành lập vào ngày tháng năm nào?"
    question_input = st.text_input("Nhập câu hỏi:", value=sample_question)
    
    # Sample questions
    st.markdown("### Câu hỏi mẫu:")
    sample_questions = [
        "Trường Đại học Công nghệ Thông tin được thành lập vào ngày tháng năm nào?",
        "Trường Đại học Công nghệ Thông tin là thành viên của hệ thống đại học nào?",
        "Trường có nhiệm vụ gì trong lĩnh vực công nghệ thông tin?"
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"btn_{q[:30]}"):
            question_input = q
    
    submit_btn = st.button("🚀 Trích xuất Câu trả lời", type="primary")
    
    # Model info
    with st.expander("ℹ️ Thông tin mô hình"):
        st.markdown(f"""
        **CS116 - Đồ án T11: Đọc hiểu máy trích xuất tiếng Việt**
        
        - **Mô hình đang chọn:** {model_choice}
        - **Dataset:** UIT-ViQuAD 2.0
        - **Nhóm 7:** Trần Trọng Tấn, Nguyễn Quang Lâm
        - **Giảng viên hướng dẫn:** ThS. Nguyễn Hữu Quyền
        
        *Lưu ý: Mô hình PhoBERT cần kết nối internet để tải về lần đầu.*
        """)

with col2:
    st.subheader("🎯 Kết quả Trích xuất (Extracted Answer)")
    if submit_btn and context_input and question_input:
        with st.spinner("Đang xử lý..."):
            answer = qa_engine.predict(context_input, question_input)
        
        st.success(f"**Câu trả lời:** {answer}")
        
        # Highlight answer in context
        if answer and len(answer) > 2:
            try:
                start_idx = context_input.find(answer)
                if start_idx != -1:
                    end_idx = start_idx + len(answer)
                    highlighted = (
                        context_input[:start_idx] +
                        f"<mark style='background: #fde047; padding: 2px 4px; border-radius: 4px; font-weight: bold;'>{answer}</mark>" +
                        context_input[end_idx:]
                    )
                    st.markdown(f"**Vị trí trong ngữ cảnh:**<br>{highlighted}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Ngữ cảnh đầy đủ:**\n{context_input}")
            except Exception as e:
                st.markdown(f"**Ngữ cảnh đầy đủ:**\n{context_input}")
        else:
            st.markdown(f"**Ngữ cảnh đầy đủ:**\n{context_input}")
        
        st.info("💡 **Mô hình triển khai:** Transformer PhoBERT-base QA fine-tuned on UIT-ViQuAD.")
        
        # Model comparison metrics (if available)
        with st.expander("📊 So sánh mô hình"):
            st.markdown("""
            | Mô hình | Exact Match | F1-Score | Tốc độ |
            |---------|-------------|----------|--------|
            | TF-IDF Baseline | ~25% | ~35% | Nhanh |
            | PhoBERT-base-v2 | ~62% | ~74% | Chậm |
            | ViDeBERTa-base | ~65% | ~77% | Chậm |
            """)
    else:
        st.info("Nhập ngữ cảnh và câu hỏi, sau đó nhấn nút để nhận kết quả!")