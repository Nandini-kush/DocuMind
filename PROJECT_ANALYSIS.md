# DocuMind - Technical Project Analysis Report

**Generated Date**: July 22, 2026  
**Project Name**: DocuMind (AI-Powered Document Chatbot)  
**Analysis Scope**: Full Codebase Audit & Architectural Evaluation  

---

## Executive Summary

DocuMind is an AI-powered RAG (Retrieval-Augmented Generation) document chatbot built with **FastAPI**, **Sentence Transformers**, **FAISS**, **PaddleOCR**, and **FLAN-T5**. It enables users to upload text-based PDFs, scanned PDFs, and image files (.png, .jpg, .jpeg), perform OCR extraction when necessary, chunk and vector-index text, and answer user queries based strictly on uploaded document context.

This document provides a comprehensive, non-destructive audit of the entire codebase as requested.

---

## 1. Folder Structure

```
DocuMind/
│
├── Agents.md                      # Project guidelines, technical specifications & feature goals
├── DocuMind_Interview_Guide.md    # Comprehensive technical interview preparation document
├── README.md                      # Setup instructions, architecture summary, Docker guide
├── .gitignore                     # Git exclusion rules
│
└── backend/                       # Python FastAPI Backend Workspace
    ├── Dockerfile                 # Debian Python 3.10-slim container image specification
    ├── main.py                    # FastAPI entrypoint, CORS configuration, API endpoints & RAG pipeline
    ├── requirements.txt           # Python dependency locks
    ├── test.py                    # Virtual environment sanity script
    │
    ├── services/
    │   └── ocr_service.py         # PaddleOCR integration, PyMuPDF page rendering, OCR routing
    │
    ├── utils/
    │   ├── llm.py                 # HuggingFace FLAN-T5 pipeline & prompt formatting
    │   ├── pdf_reader.py          # PyPDF2 text extraction module
    │   ├── text_chunker.py        # Text regex cleaning & overlapping word-chunking
    │   └── vector_store.py        # SentenceTransformer embedding & FAISS index management
    │
    ├── tests/
    │   └── test_ocr.py            # Pytest test suite for OCR detection heuristics and unit mocks
    │
    └── uploads/                   # Runtime local storage directory for uploaded documents
```

---

## 2. Technologies Used

### Backend Core
* **Python**: `3.10`
* **Web Framework**: `FastAPI` (v0.124.4) with `Uvicorn` (v0.38.0) & `Pydantic` (v2.12.5)
* **ASGI Server & Middleware**: `Starlette` (v0.50.0) with `CORSMiddleware`

### AI & Vector Processing
* **Embeddings**: `sentence-transformers` (v5.2.0) using model `all-MiniLM-L6-v2` (384-dimensional dense vectors)
* **Vector Store**: `faiss-cpu` (v1.13.1) using L2 Euclidean distance index (`IndexFlatL2`)
* **LLM Engine**: `transformers` (v4.57.3) with `torch` (v2.9.1) running `google/flan-t5-base` (`text2text-generation` pipeline)

### Document Parsing & OCR
* **PDF Reader**: `PyPDF2` (v3.0.1) for native selectable text extraction
* **PDF-to-Image Converter**: `PyMuPDF` (`fitz`) rendering pages at 200 DPI
* **OCR Engine**: `PaddleOCR` / `paddlepaddle` with angle classification enabled (`use_angle_cls=True`)
* **Image Libraries**: `Pillow`, `pdf2image`, `scikit-learn`, `scipy`

### Infrastructure & Deployment
* **Containerization**: Docker (`python:3.10-slim`)
* **System Dependencies**: `poppler-utils`, `libgl1`, `libglib2.0-0`, `libgomp1`

---

## 3. Backend Architecture

The backend follows a modular service-utility architecture around FastAPI:

```mermaid
flowchart TD
    Client[Client / REST API Request] --> FastAPI[main.py - FastAPI Application]
    FastAPI --> Upload[/upload Endpoint]
    FastAPI --> RAG[/rag Endpoint]
    
    Upload --> PDFReader[utils/pdf_reader.py]
    PDFReader -- Text < 500 chars --> OCR[services/ocr_service.py]
    PDFReader -- Text >= 500 chars --> Chunker[utils/text_chunker.py]
    OCR --> Chunker
    Chunker --> Embedder[utils/vector_store.py]
    Embedder --> FAISS[(In-Memory FAISS Index & Chunks)]
    
    RAG --> FAISS
    FAISS -- Top-K Chunks --> Filter[Noise & Keyword Filtering]
    Filter --> LLM[utils/llm.py - FLAN-T5]
    LLM --> Response[JSON Answer + Sources]
```

### Key Architectural Characteristics
1. **Single-Tenant / In-Memory State**: Global variables `faiss_index` and `stored_chunks` in `main.py` hold the current document index in memory.
2. **Lazy Initialization**: Models (`SentenceTransformer`, `PaddleOCR`, `FLAN-T5`) are loaded at backend startup to minimize inference delay per request.
3. **Synchronous Inference**: RAG operations run synchronously per endpoint request.

---

## 4. Frontend Architecture

### Status: **Missing / Planned**
* **Documentation Specifications**: `Agents.md` specifies a React.js client with Axios and custom CSS. `README.md` outlines Next.js + Tailwind CSS + Framer Motion.
* **Repository State**: No source files for the frontend exist in the project directory. Currently, API endpoints can only be interacted with via FastAPI `/docs` (Swagger UI) or Postman/cURL.

---

## 5. OCR Workflow

Detailed in `backend/services/ocr_service.py`:

1. **Trigger Heuristic (`should_use_ocr`)**:
   * Evaluates text extracted by PyPDF2.
   * If text is empty or contains `< 500` characters, triggers OCR.
   * Images (`.png`, `.jpg`, `.jpeg`) automatically bypass PyPDF2 and route straight to OCR.
2. **PDF Page Rendering**:
   * PyMuPDF (`fitz`) opens the PDF and renders each page to a temporary 200 DPI PNG file (`{pdf_path}_page_{page_num}.png`).
3. **PaddleOCR Text Extraction**:
   * Passes each image to `PaddleOCR.ocr(image_path, cls=True)`.
   * Aggregates extracted lines into text strings per page.
4. **Cleanup & Structuring**:
   * Temporary page PNGs are deleted in a `finally:` block.
   * Returns structured output: `[{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]`.

---

## 6. RAG Workflow

Detailed in `backend/main.py` (`/rag` endpoint):

1. **Query Encoding**: Converts user query string into a 384-d vector via `SentenceTransformer`.
2. **FAISS Distance Search**: Retrieves top $K=2$ most similar chunks. Computes similarity score $S = \frac{1}{1 + \text{L2\_distance}}$.
3. **Hard Distance Filtering**: Discards chunks with similarity score $< 0.35$.
4. **Noise Removal (`is_noise`)**:
   * Discards chunks with length $< 40$ characters.
   * Discards chunks with uppercase word ratio $> 0.8$.
   * Discards chunks with digit ratio $> 0.5$.
5. **Keyword Relevance Re-ranking**:
   * Scores remaining chunks by counting overlapping words between user query and chunk text.
   * Sorts descending and selects **only the top 1 chunk** (`[:1]`).
   * Falls back to vector top match if no keyword overlap exists.
6. **Prompt Formatting & LLM Inference**:
   * Truncates context to 3,000 characters.
   * Prompts `FLAN-T5` with strict context constraints.
   * Evaluates response: if empty or indicates missing info, returns `"The document does not contain this information."`.

---

## 7. Embedding Pipeline

Detailed in `backend/utils/vector_store.py`:

* **Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Dimension**: 384 dimensions
* **Input**: Plain text string extracted from each chunk dict (`chunk["text"]`).
* **Output**: Dense 2D numpy array (`shape: [N, 384]`).
* **Progress Tracking**: `show_progress_bar=False` for quiet server execution.

---

## 8. FAISS Storage

* **Index Type**: `faiss.IndexFlatL2(384)` (Euclidean L2 distance matrix).
* **Storage Location**: Volatile RAM (`faiss_index` global variable in `main.py`).
* **Index Life Cycle**: Created dynamically on file upload via `create_faiss_index()`.
* **Persistence**: **None**. Index resets upon server restart or new document upload.

---

## 9. LLM Workflow

Detailed in `backend/utils/llm.py`:

* **Model**: `google/flan-t5-base`
* **Framework**: HuggingFace `pipeline("text2text-generation")`
* **Execution Target**: Auto-detects CUDA GPU (`device=0`) or falls back to CPU (`device=-1`).
* **Decoding Parameters**: `max_new_tokens=100`, `do_sample=False` (deterministic greedy decoding).
* **System Prompt Template**:
  ```text
  Answer the question based only on the following context. If the answer is not contained in the context, output exactly 'The document does not contain this information.'

  Context:
  {context}

  Question: {question}

  Answer:
  ```
* **Guardrails**: If generated answer is $< 2$ characters or contains hallucination patterns, returns standard fallback string.

---

## 10. API Endpoints

| Method | Endpoint | Description | Request Body | Response Format |
|---|---|---|---|---|
| `GET` | `/` | Health Check | None | `{"status": "FastAPI is running"}` |
| `POST` | `/upload` | PDF / Image Ingestion | `multipart/form-data` (`file`) | `{"filename": str, "total_chunks": int, "embedding_dimension": int}` |
| `POST` | `/search` | Semantic Vector Search | `{"query": str}` | `{"query": str, "top_matches": [{"text": str, "score": float}]}` |
| `POST` | `/rag` | Question Answering | `{"query": str}` | `{"question": str, "answer": str, "sources": [str]}` |

---

## 11. Current Bugs & Issues

> [!WARNING]
> **Critical Architectural & Code Issues Identified:**

1. **Global State Concurrency Overwrite Bug**:
   * `faiss_index` and `stored_chunks` are stored in global variables (`main.py` L25-26).
   * Uploading a new document completely overwrites the active index for *all* connected users.
2. **Strict Single-Chunk Truncation Negates Vector Search**:
   * `main.py` lines 203-206 restrict context to `scored_chunks[:1]` (a single chunk) based on exact word matching.
   * If a user query uses synonyms not present in the document text, exact word match score is 0, degrading RAG accuracy.
3. **Aggressive Noise Filter (`is_noise`) Data Loss**:
   * `is_noise` discards any chunk with numeric ratio $> 0.5$ or character length $< 40$.
   * This accidentally drops financial tables, marksheets, phone numbers, addresses, and short concise facts.
4. **Temporary File Collisions in OCR**:
   * `temp_image_path = f"{pdf_path}_page_{page_num}.png"` is deterministic. Concurrent uploads of files with matching names will collide and corrupt OCR processing.
5. **Unbounded Disk Storage**:
   * Uploaded files are copied directly into `uploads/` without file size limits, rate limiting, or periodic cleanup routines.

---

## 12. Missing Features

1. **Frontend Interface**: No UI components or React/Next.js codebase present in project.
2. **User Authentication & Multi-Tenancy**: Lacks JWT/OAuth authentication and user session isolation.
3. **Persistent Vector Database**: Lacks persistent vector storage (e.g., ChromaDB, Pinecone, or disk-backed FAISS).
4. **Multi-Document Support**: Lacks ability to manage, search, or switch between multiple uploaded documents.
5. **Document Format Support**: Lacks support for `.docx`, `.txt`, `.csv`, `.pptx`.
6. **Streaming API Responses**: Lacks WebSocket or Server-Sent Events (SSE) streaming for LLM outputs.

---

## 13. Deployment Readiness

| Category | Rating | Notes |
|---|---|---|
| **Containerization** | 🟡 Partial | Dockerfile is well structured with system dependencies (`poppler-utils`, `libgl1`), but lacks multi-stage builds. |
| **Multi-User Scalability** | 🔴 Not Ready | Global in-memory variables prevent concurrent user isolation. |
| **Resource Efficiency** | 🟡 Demanding | Running PaddleOCR + FLAN-T5 on CPU requires high RAM and CPU allocation. |
| **Security & Auth** | 🔴 Missing | No API authentication, rate limiting, or input file sanitization. |
| **Production Grade** | 🔴 Not Ready | Requires persistent vector DB, session management, and a web frontend before deployment. |

---

## Conclusion & Recommended Next Steps

DocuMind demonstrates a solid conceptual foundation for an OCR-enabled RAG pipeline. To transition it to a production-ready application, the following refactoring roadmap is recommended:

1. **Implement Multi-Tenant Architecture**: Replace global memory variables with session-based vector stores or ChromaDB persistent storage.
2. **Build the Frontend Client**: Develop the React/Next.js dashboard as outlined in `README.md`.
3. **Optimize RAG Retrieval Pipeline**: Relax strict keyword-matching filters and allow multi-chunk context synthesis for FLAN-T5.
4. **Add Cleanup Routines**: Implement automatic purging of temporary upload files and UUID-based temporary file naming.
