from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import shutil
import os

from utils.pdf_reader import extract_text_from_pdf
from utils.text_chunker import clean_text, chunk_text
from utils.vector_store import create_faiss_index, search_similar_chunks
from utils.llm import generate_answer


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- GLOBAL STORAGE ----------------
faiss_index = None
stored_chunks = []


# ---------------- SCHEMAS ----------------
class Question(BaseModel):
    query: str


# ---------------- HEALTH CHECK ----------------
@app.get("/")
def health_check():
    return {"status": "FastAPI is running"}


# ---------------- UPLOAD PDF ----------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global faiss_index, stored_chunks

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract → clean → chunk
    raw_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(raw_text)

    # IMPORTANT: chunk_text MUST return dict-based chunks
    chunks = chunk_text(cleaned_text)

    if not chunks:
        return {"error": "No text chunks generated from PDF"}

    index, embeddings = create_faiss_index(chunks)

    faiss_index = index
    stored_chunks = chunks

    return {
        "filename": file.filename,
        "total_chunks": len(chunks),
        "embedding_dimension": embeddings.shape[1]
    }


# ---------------- SEMANTIC SEARCH (OPTIONAL) ----------------
@app.post("/search")
def semantic_search(data: Question):
    if faiss_index is None:
        return {"error": "No document uploaded yet"}

    results = search_similar_chunks(
        query=data.query,
        chunks=stored_chunks,
        index=faiss_index,
        top_k=5
    )

    return {
        "query": data.query,
        "top_matches": results
    }


# ---------------- RAG QUESTION ANSWERING ----------------
@app.post("/rag")
def rag_answer(data: Question):
    if faiss_index is None:
        return {"error": "No document uploaded yet"}

    retrieved_chunks = search_similar_chunks(
        query=data.query,
        chunks=stored_chunks,
        index=faiss_index,
        top_k=4
    )

    if not retrieved_chunks:
        return {"error": "No relevant context found"}

    # ---------------- STEP 1: REMOVE NOISE ----------------
    def is_noise(text: str) -> bool:
        text = text.strip()

        if len(text) < 60:
            return True

        words = text.split()
        if not words:
            return True

        capital_ratio = sum(1 for w in words if w.isupper()) / len(words)
        if capital_ratio > 0.4:
            return True

        return False

    clean_chunks = [
        chunk for chunk in retrieved_chunks
        if not is_noise(chunk["text"])
    ]

    if not clean_chunks:
        clean_chunks = retrieved_chunks

    # ---------------- STEP 2: QUERY RELEVANCE FILTER ----------------
    query_words = set(data.query.lower().split())

    scored_chunks = []
    for chunk in clean_chunks:
        text = chunk["text"].lower()
        score = sum(1 for word in query_words if word in text)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # keep ONLY the most relevant chunks
    filtered_chunks = [
        chunk for score, chunk in scored_chunks
        if score > 0
    ][:2]

    # fallback
    if not filtered_chunks:
        filtered_chunks = [chunk for _, chunk in scored_chunks[:2]]

    # ---------------- DEBUG ----------------
    print("\nRetrieved Chunks Sent to LLM:\n")
    for i, chunk in enumerate(filtered_chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk["text"][:300])
        print("\n")

    # ---------------- LLM ----------------
    context_texts = [chunk["text"] for chunk in filtered_chunks]

    answer = generate_answer(
        question=data.query,
        context_chunks=context_texts
    )

    return {
        "question": data.query,
        "answer": answer,
        "sources": context_texts
    }