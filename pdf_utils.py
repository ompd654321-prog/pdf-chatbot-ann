import PyPDF2

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "

    return text


# Improved sentence-based chunking
def chunk_text(text, chunk_size=5):
    sentences = text.split(". ")
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)

    return chunks