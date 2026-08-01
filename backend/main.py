import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.pdf_reader import extract_text_from_pdf
from services.ocr_service import extract_text_with_ocr, extract_text_from_image_bytes, should_use_ocr
from utils.text_chunker import clean_text, chunk_text, deduplicate_chunks
from utils.vector_store import generate_embeddings, create_faiss_index, search_vector_store, DEFAULT_SIMILARITY_THRESHOLD
from utils.llm import generate_answer, FALLBACK_RESPONSE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind API",
    description="AI-powered Document Question Answering System with OCR and RAG",
    version="1.0.0"
)

FRONTEND_URLS = os.getenv("FRONTEND_URLS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global In-Memory Vector Storage State
document_chunks: List[str] = []
faiss_index = None
active_document_name: Optional[str] = None


class RAGQueryRequest(BaseModel):
    query: str


class SearchQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


@app.get("/")
def health_check():
    """Health check endpoint providing active backend status and vector store stats."""
    return {
        "status": "FastAPI is running",
        "has_document": faiss_index is not None and len(document_chunks) > 0,
        "active_document": active_document_name,
        "total_chunks": len(document_chunks)
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Handles document ingestion (PDF, PNG, JPG, JPEG), text extraction, chunking, and FAISS indexing."""
    global document_chunks, faiss_index, active_document_name

    logger.info(f"[API-UPLOAD] Received upload request for file: {file.filename}")

    # Validate file extension
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        logger.warning(f"[API-UPLOAD] Rejected file format: '{ext}'")
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF, PNG, JPG, or JPEG file."
        )

    # Save to disk with unique UUID
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        content = await file.read()
        if len(content) == 0:
            logger.warning("[API-UPLOAD] Empty 0-byte file received.")
            raise HTTPException(status_code=422, detail="Failed to extract text from document.")

        with open(file_path, "wb") as f:
            f.write(content)

        extracted_text = ""

        # Processing PDF vs Image
        if ext == "pdf":
            logger.info(f"[API-UPLOAD] Extracting native PDF text using PyMuPDF...")
            raw_text = extract_text_from_pdf(file_path)

            if should_use_ocr(raw_text, min_chars=50):
                logger.info("[API-UPLOAD] Native PDF text < 50 chars. Triggering PaddleOCR fallback...")
                res = extract_text_with_ocr(file_path)
                extracted_text = str(res)
                if not extracted_text.strip() and raw_text.strip():
                    logger.info("[API-UPLOAD] OCR returned empty result. Retaining native PDF text.")
                    extracted_text = raw_text
            else:
                logger.info(f"[API-UPLOAD] Native PDF text extracted successfully ({len(raw_text)} chars).")
                extracted_text = raw_text
        else:
            logger.info(f"[API-UPLOAD] Ingesting image file (.png/.jpg/.jpeg) via PaddleOCR...")
            extracted_text = extract_text_from_image_bytes(content)

        # Cleanup temporary upload file
        if os.path.exists(file_path):
            os.remove(file_path)

        cleaned = clean_text(str(extracted_text))
        if not cleaned:
            logger.warning("[API-UPLOAD] Text cleaning produced zero characters.")
            raise HTTPException(status_code=422, detail="Failed to extract text from document.")

        # Chunking & Deduplication (Default: 500 chars, 100 overlap)
        raw_chunks = chunk_text(cleaned, chunk_size=500, overlap=100)
        unique_chunks = deduplicate_chunks(raw_chunks)

        if not unique_chunks:
            logger.warning("[API-UPLOAD] Chunking produced zero valid chunks.")
            raise HTTPException(status_code=422, detail="Failed to extract text from document.")

        logger.info(f"[API-UPLOAD] Generated {len(unique_chunks)} unique chunks from {len(raw_chunks)} raw chunks.")

        # Generate Embeddings & Create FAISS Index
        new_faiss_index, embeddings = create_faiss_index(unique_chunks)

        # Atomic State Update
        document_chunks = unique_chunks
        faiss_index = new_faiss_index
        active_document_name = file.filename

        logger.info(f"[API-UPLOAD] Successfully indexed document '{file.filename}' into FAISS with {len(unique_chunks)} chunks.")

        return {
            "filename": file.filename,
            "status": "success",
            "total_chunks": len(unique_chunks),
            "embedding_dimension": embeddings.shape[1] if embeddings.ndim == 2 else 384
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API-UPLOAD] Unexpected error during upload: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/rag")
async def rag_query(request: RAGQueryRequest):
    """Executes RAG pipeline: Query Embedding -> FAISS Vector Search -> Hybrid Re-rank -> FLAN-T5 Generation."""
    global document_chunks, faiss_index

    query_text = request.query.strip() if request.query else ""
    logger.info(f"[API-RAG] Query received: '{query_text}'")

    if not query_text:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    if faiss_index is None or not document_chunks:
        logger.info("[API-RAG] No active document index present. Returning fallback response.")
        return {
            "question": query_text,
            "answer": FALLBACK_RESPONSE,
            "sources": []
        }

    try:
        # Search vector store & hybrid re-rank with 0.15 threshold and intelligent fallback
        results = search_vector_store(
            query=query_text,
            index=faiss_index,
            chunks=document_chunks,
            top_k=5,
            similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD
        )

        if not results:
            logger.info("[API-RAG] No context chunks available for LLM inference.")
            return {
                "question": query_text,
                "answer": FALLBACK_RESPONSE,
                "sources": []
            }

        top_chunks = [chunk["text"] if isinstance(chunk, dict) else str(chunk) for chunk in results[:3]]

        # Generate extractive answer
        res = generate_answer(query_text, top_chunks)
        answer = str(res)
        sources = getattr(res, "sources", [f"[Source {i+1}]: {c}" for i, c in enumerate(top_chunks)])

        return {
            "question": query_text,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"[API-RAG] Unexpected error during RAG query: {e}")
        return {
            "question": query_text,
            "answer": FALLBACK_RESPONSE,
            "sources": []
        }


@app.post("/search")
async def search_chunks(request: SearchQueryRequest):
    """Semantic vector search endpoint returning raw chunk context and scores."""
    global document_chunks, faiss_index

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    if faiss_index is None or not document_chunks:
        return {"query": request.query, "results": []}

    try:
        results = search_vector_store(
            query=request.query.strip(),
            index=faiss_index,
            chunks=document_chunks,
            top_k=request.top_k or 5,
            similarity_threshold=0.0
        )

        formatted_results = [
            {"chunk": item.get("text", str(item)), "similarity_score": round(float(item.get("score", 0.0)), 4)}
            for item in results
        ]

        return {
            "query": request.query,
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"[API-SEARCH] Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))