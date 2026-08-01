import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_chunker import clean_text, chunk_text, deduplicate_chunks
from utils.vector_store import create_faiss_index, search_similar_chunks
from main import app

client = TestClient(app)

def test_clean_text():
    raw = "  Page 1 of 5 \n  Hello    World!   \n\n\n  DocuMind RAG test.  "
    cleaned = clean_text(raw)
    assert "Page 1" not in cleaned
    assert "Hello World!" in cleaned
    assert "DocuMind RAG test." in cleaned

def test_deduplicate_chunks():
    chunks = [
        {"chunk_id": 0, "text": "Duplicate sentence text here."},
        {"chunk_id": 1, "text": "Duplicate sentence text here."},
        {"chunk_id": 2, "text": "Unique sentence text here."}
    ]
    unique = deduplicate_chunks(chunks)
    assert len(unique) == 2
    texts = [c["text"] for c in unique]
    assert "Duplicate sentence text here." in texts
    assert "Unique sentence text here." in texts

def test_vector_store_cosine_similarity():
    chunks = [
        {"chunk_id": 0, "text": "Python fast API web application for artificial intelligence."},
        {"chunk_id": 1, "text": "DocuMind uses FAISS vector embeddings and sentence transformers."},
        {"chunk_id": 2, "text": "Cooking recipe for chocolate chip pancakes."}
    ]
    index, embeddings = create_faiss_index(chunks)
    assert index is not None
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 384

    # Search query
    results = search_similar_chunks("FAISS vector embeddings", chunks, index, top_k=2)
    assert len(results) >= 1
    assert "FAISS vector embeddings" in results[0]["text"]
    assert results[0]["score"] > 0.40

def test_health_check_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "FastAPI is running"
