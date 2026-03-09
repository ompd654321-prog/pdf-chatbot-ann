import streamlit as st
from pdf_utils import extract_text_from_pdf, chunk_text
from embedding_utils import create_embeddings, store_embeddings, get_answer

st.set_page_config(page_title="PDF Chatbot ANN", layout="wide")

st.title("🤖 PDF Chatbot using ANN")
st.write("Ask questions strictly from the uploaded PDF.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file and st.session_state.index is None:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        index = store_embeddings(embeddings)

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success("PDF processed successfully!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question from the PDF..."):

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.index is not None:
        with st.spinner("Thinking..."):
            answers, scores = get_answer(
                prompt,
                st.session_state.index,
                st.session_state.chunks
            )

            response = answers[0]

        # Store bot response
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        with st.chat_message("assistant"):
            st.markdown(response)

    else:
        st.warning("Please upload a PDF first.")
