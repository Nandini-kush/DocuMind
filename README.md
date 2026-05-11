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
