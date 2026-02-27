import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load Transformer-based ANN model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


def store_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


# Improved answer function (Top 3 + similarity score)
def get_answer(question, index, chunks):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), k=3)

    results = []
    scores = []

    for idx, dist in zip(I[0], D[0]):
        results.append(chunks[idx])
        scores.append(dist)

    return results, scores