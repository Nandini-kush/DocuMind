<<<<<<< HEAD
# DocuMind – AI-Powered PDF Chatbot

DocuMind is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDF documents and ask natural language questions. The system retrieves relevant content from the document and generates accurate answers using AI.

## 🚀 Features
- Upload and analyze PDF documents
- Semantic search using FAISS
- RAG-based question answering
- Modern React chatbot interface
- Accurate, document-grounded responses

## 🛠️ Tech Stack
**Frontend:** React.js, Axios, CSS  
**Backend:** FastAPI, Python  
**AI/ML:** Sentence Transformers, FAISS, FLAN-T5, RAG

## 📂 Project Structure
```text
DocuMind/
├── backend/
├── frontend/
├── .gitignore
└── README.md

Backend:
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload

Frontend:
cd frontend
npm install
npm start

API Endpoints:
POST /upload – Upload PDF
POST /search – Semantic search
POST /rag – Ask questions from the document
=======
# DocuMind - AI-Powered Document Chatbot with OCR & RAG

**DocuMind** is a production-grade, AI-powered document chatbot built with **FastAPI**, **Sentence Transformers**, **FAISS**, **PaddleOCR**, **FLAN-T5**, and **React.js**. It enables users to upload native PDFs, scanned PDFs, and image files (.png, .jpg, .jpeg), perform OCR extraction when necessary, chunk and vector-index document text using Cosine Similarity, and answer user queries with zero hallucination and explicit source attribution.

---

## 🌟 Key Features

* **Multi-Format Document Ingestion**: Native PDF parsing via PyMuPDF (`fitz`), scanned PDF fallback OCR, and image processing (`.png`, `.jpg`, `.jpeg`).
* **In-Memory PaddleOCR Engine**: High-DPI image preprocessing (grayscale, contrast boost, sharpness optimization) and confidence filtering ($\ge 0.50$).
* **Normalized Cosine Vector Search**: `SentenceTransformer("all-MiniLM-L6-v2")` generating 384-dimensional dense vectors indexed with FAISS `IndexFlatIP`.
* **Hybrid Re-ranking & Context Building**: Combines vector Cosine similarity ($0.7$) with keyword density ($0.3$) and passes multi-chunk context (`[Source 1]`, `[Source 2]`) to the LLM.
* **Extractive Answer Generation & Guardrails**: FLAN-T5 (`google/flan-t5-base`) with strict fallback rules. If information is unavailable, guarantees exact response: `"The document does not contain this information."`
* **Modern Glassmorphic React UI**: React 18 + Vite SPA with drag-and-drop file upload, live OCR engine status badge, expandable context accordions, LocalStorage history persistence, and responsive mobile drawer.

---

## 🏗️ Architecture Blueprint

```mermaid
flowchart TD
    Client[React.js Frontend Dashboard - Port 3000] <-->|REST API / JSON| FastAPI[FastAPI Backend Server - Port 8000]
    
    FastAPI --> Upload[/upload Endpoint]
    FastAPI --> RAG[/rag Endpoint]
    
    Upload --> PDFCheck{Selectable PDF >= 50 Chars?}
    PDFCheck -- Yes --> PyMuPDF[PyMuPDF Native Text Extraction]
    PDFCheck -- No / Image --> OCR[PaddleOCR + PIL Preprocessing]
    
    PyMuPDF --> Chunker[Boundary Word Chunker + Hash Deduplication]
    OCR --> Chunker
    
    Chunker --> Embedder[SentenceTransformer all-MiniLM-L6-v2]
    Embedder --> FAISS[(FAISS IndexFlatIP Cosine Index)]
    
    RAG --> FAISS
    FAISS -- Cosine Score >= 0.30 --> ReRanker[Hybrid Re-ranker]
    ReRanker --> ExtractivePrompt[Extractive Prompt Template]
    ExtractivePrompt --> LLM[HuggingFace FLAN-T5 Model]
    LLM --> JSONResponse[Answer + Sources Output]
```

---

## 💻 Local Quickstart Guide

### Prerequisites
* **Python**: 3.10 through 3.13
* **Node.js**: v18+ and `npm`
* **System Dependencies** (for OCR & PDF processing):
  - Ubuntu/Debian: `sudo apt-get install poppler-utils libgl1 libglib2.0-0 libgomp1`
  - macOS: `brew install poppler`
  - Windows: Handled automatically by Python binaries.

### 1. Run the Backend
```bash
# Navigate to backend directory
cd backend

# Create & activate virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend API will be live at: `http://localhost:8000` (Swagger UI at `http://localhost:8000/docs`).

### 2. Run the Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Frontend Web Dashboard will be live at: `http://localhost:3000`.

---

## 🐳 Docker Deployment Guide

To run the backend inside a production Docker container:

```bash
# Build the Docker image
cd backend
docker build -t documind-backend .

# Run the container
docker run -d -p 8000:8000 --name documind-backend-container documind-backend
```

---

## ☁️ Cloud Deployment Guide (Render / PaaS)

This repository includes a `render.yaml` blueprint for one-click deployment:

1. Push your repository to **GitHub**.
2. Log into [Render.com](https://render.com) and click **Blueprints** $\rightarrow$ **New Blueprint Instance**.
3. Select your `DocuMind` repository. Render automatically reads `render.yaml` and deploys:
   - **`documind-backend`**: Docker Web Service running FastAPI on port 8000.
   - **`documind-frontend`**: Static Web Site serving Vite build output (`frontend/dist`).

---

## 🧪 Automated Test Suite

To run the complete automated test suite (24 tests across LLM, OCR, RAG, and E2E scenarios):

```bash
python -m pytest backend/tests
```

---

## 📄 License & Attribution

Distributed under the MIT License. Built for production document intelligence.
>>>>>>> 7954cbd (final changes done)
