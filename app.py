import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Load summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit UI
st.title("📄 Smart PDF Summarizer")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    st.subheader("Extracted Text Sample:")
    st.write(text[:500])

    # Summarize
    if st.button("Generate Summary"):
        summary = summarizer(text[:3000], max_length=150, min_length=50, do_sample=False)
        final_summary = summary[0]['summary_text']
        
        st.subheader("Summary:")
        st.write(final_summary)

        # Save to summary.txt
        with open("summary.txt", "w", encoding="utf-8") as f:
            f.write(final_summary)

        st.success("✅ Summary saved to summary.txt")
        # Download button
        st.download_button(
            label="📥 Download Summary",
            data=final_summary,
            file_name="summary.txt",
            mime="text/plain"
        )

