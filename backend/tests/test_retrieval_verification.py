import sys
import os
import io
import pytest
import fitz  # PyMuPDF
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)


def create_sample_resume_pdf_bytes() -> bytes:
    """Creates a sample candidate resume PDF bytes for testing."""
    resume_text = (
        "Nandini Kushwah\n"
        "Email: nandini.kushwah@example.com | Phone: +91-9876543210 | Location: India\n"
        "GitHub: github.com/nandinikushwah | LinkedIn: linkedin.com/in/nandinikushwah\n\n"
        "--- EDUCATION ---\n"
        "Bachelor of Technology in Computer Science and Engineering\n"
        "National Institute of Technology | Graduation Year: 2025 | CGPA: 8.95 / 10.0\n\n"
        "--- TECHNICAL SKILLS ---\n"
        "Programming Languages: Python, JavaScript, C++, SQL, HTML, CSS\n"
        "Frameworks & Libraries: FastAPI, React.js, PyTorch, Transformers, SentenceTransformers, FAISS, PyMuPDF, PaddleOCR\n"
        "Tools & Platforms: Git, Docker, Linux, HuggingFace Hub, VS Code\n\n"
        "--- WORK EXPERIENCE & INTERNSHIPS ---\n"
        "Machine Learning Research Intern | AI Innovation Labs (Summer 2024)\n"
        "- Developed semantic vector retrieval pipelines using SentenceTransformers and FAISS IndexFlatIP.\n"
        "- Implemented Reciprocal Rank Fusion (RRF) and CrossEncoder reranking algorithms to optimize search precision.\n"
        "- Improved document query accuracy by 45% across complex PDF text extraction scenarios.\n\n"
        "--- PROJECTS ---\n"
        "1. DocuMind - AI Document Chatbot & RAG System\n"
        "- Built a full-stack Retrieval-Augmented Generation application using FastAPI, React, and FLAN-T5.\n"
        "- Integrated PyMuPDF and PaddleOCR for native and scanned PDF text extraction.\n"
        "- Features section-aware recursive chunking, BM25 sparse search, and FAISS dense vector indexing.\n\n"
        "2. YojnaSaathi - Government Scheme Discovery Portal\n"
        "- Developed a web application for discovering welfare schemes tailored to user demographics.\n"
        "- Created intelligent search indexing using multi-lingual NLP models.\n"
    )

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), resume_text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_upload_resume_document():
    """Uploads sample resume PDF and confirms indexing success."""
    pdf_bytes = create_sample_resume_pdf_bytes()
    response = client.post(
        "/upload",
        files={"file": ("Nandini_Kushwah_Resume.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "Nandini_Kushwah_Resume.pdf"
    assert data["status"] == "success"
    assert data["total_chunks"] >= 1


def test_query_1_candidate_name():
    """Target Query 1: Candidate Name Retrieval"""
    pdf_bytes = create_sample_resume_pdf_bytes()
    client.post("/upload", files={"file": ("Nandini_Kushwah_Resume.pdf", pdf_bytes, "application/pdf")})

    response = client.post("/rag", json={"query": "What is the candidate name?"})
    assert response.status_code == 200
    data = response.json()
    assert "Nandini" in data["answer"] or "Nandini Kushwah" in data["answer"] or "Nandini" in str(data["sources"])


def test_query_2_email():
    """Target Query 2: Candidate Email Retrieval"""
    pdf_bytes = create_sample_resume_pdf_bytes()
    client.post("/upload", files={"file": ("Nandini_Kushwah_Resume.pdf", pdf_bytes, "application/pdf")})

    response = client.post("/rag", json={"query": "What is the email?"})
    assert response.status_code == 200
    data = response.json()
    assert "nandini.kushwah@example.com" in data["answer"] or "nandini" in data["answer"] or "example.com" in str(data["sources"])


def test_query_3_cgpa():
    """Target Query 3: Candidate CGPA Retrieval"""
    pdf_bytes = create_sample_resume_pdf_bytes()
    client.post("/upload", files={"file": ("Nandini_Kushwah_Resume.pdf", pdf_bytes, "application/pdf")})

    response = client.post("/rag", json={"query": "What is the CGPA?"})
    assert response.status_code == 200
    data = response.json()
    assert "8.95" in data["answer"] or "8.95" in str(data["sources"])


def test_query_4_projects():
    """Target Query 4: Projects Retrieval"""
    pdf_bytes = create_sample_resume_pdf_bytes()
    client.post("/upload", files={"file": ("Nandini_Kushwah_Resume.pdf", pdf_bytes, "application/pdf")})

    response = client.post("/rag", json={"query": "List all projects."})
    assert response.status_code == 200
    data = response.json()
    assert "DocuMind" in data["answer"] or "YojnaSaathi" in data["answer"] or "DocuMind" in str(data["sources"])


def test_query_5_programming_languages():
    """Target Query 5: Programming Languages Retrieval"""
    pdf_bytes = create_sample_resume_pdf_bytes()
    client.post("/upload", files={"file": ("Nandini_Kushwah_Resume.pdf", pdf_bytes, "application/pdf")})

    response = client.post("/rag", json={"query": "Which programming languages are known?"})
    assert response.status_code == 200
    data = response.json()
    assert "Python" in data["answer"] or "JavaScript" in data["answer"] or "Python" in str(data["sources"])


def test_query_6_internships():
    """Target Query 6: Internships Retrieval"""
    pdf_bytes = create_sample_resume_pdf_bytes()
    client.post("/upload", files={"file": ("Nandini_Kushwah_Resume.pdf", pdf_bytes, "application/pdf")})

    response = client.post("/rag", json={"query": "What internships are completed?"})
    assert response.status_code == 200
    data = response.json()
    assert "AI Innovation Labs" in data["answer"] or "Machine Learning" in data["answer"] or "Intern" in str(data["sources"])


def test_query_7_graduation_year():
    """Target Query 7: Graduation Year Retrieval"""
    pdf_bytes = create_sample_resume_pdf_bytes()
    client.post("/upload", files={"file": ("Nandini_Kushwah_Resume.pdf", pdf_bytes, "application/pdf")})

    response = client.post("/rag", json={"query": "What is the graduation year?"})
    assert response.status_code == 200
    data = response.json()
    assert "2025" in data["answer"] or "2025" in str(data["sources"])
