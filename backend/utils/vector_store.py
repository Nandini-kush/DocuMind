from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model once (good practice)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_faiss_index(chunks):
    """
    chunks: list of dicts
    [
      {"section": "...", "text": "..."},
      ...
    ]
    """

    if not chunks:
        raise ValueError("No chunks provided for embedding")

    # Embed ONLY text
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    if embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings shape: {embeddings.shape}")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


def search_similar_chunks(query, chunks, index, top_k=5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append({
                "text": chunks[idx]["text"],
                "score": float(1 / (1 + distance))  # similarity score
            })

    # sort by relevance (highest score first)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results



