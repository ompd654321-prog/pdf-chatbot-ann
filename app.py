import streamlit as st
from pdf_utils import extract_text_from_pdf, chunk_text
from embedding_utils import create_embeddings, store_embeddings, get_answer

st.set_page_config(page_title="PDF Chatbot ANN", layout="centered")

st.title("📄 PDF Chatbot using Artificial Neural Network")
st.write("This chatbot answers questions strictly from the uploaded PDF.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:

    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Creating text chunks..."):
        chunks = chunk_text(text)

    with st.spinner("Generating embeddings using ANN (first time may take 30-60 seconds)..."):
        embeddings = create_embeddings(chunks)
        index = store_embeddings(embeddings)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question from the PDF")

    if question:
        with st.spinner("Searching for answer..."):
            answers, scores = get_answer(question, index, chunks)

        st.subheader("🔎 Top Relevant Results:")

        for i in range(len(answers)):
            st.write(f"### Result {i+1}")
            st.write(answers[i])
            st.write(f"Similarity Distance: {scores[i]:.4f}")
            st.write("---")