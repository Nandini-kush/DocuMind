import sys
import os
import io
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)

# Helper generators for synthetic test files
def create_normal_pdf_bytes(text: str = "DocuMind is an AI powered document chatbot created for semantic search and question answering.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def create_scanned_pdf_bytes(text: str = "Scanned PDF Text Content") -> bytes:
    img = Image.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), text, fill="black")
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format="PNG")
    img_bytes = img_bytes_io.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=300, height=100)
    page.insert_image(page.rect, stream=img_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def create_png_image_bytes(text: str = "OCR Test Image String") -> bytes:
    img = Image.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), text, fill="black")
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format="PNG")
    return img_bytes_io.getvalue()

def create_large_pdf_bytes(num_pages: int = 10) -> bytes:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        # Generate unique content per page to prevent deduplication from merging identical pages
        unique_page_text = (
            f"Section {i+1} Chapter Overview:\n" + 
            f"This is unique paragraph block number {i+1} detailing architectural specifications, "
            f"database schema, API endpoints, vector indexing, and RAG configuration details for module {i+1}.\n"
        ) * 10
        page.insert_text((50, 50), unique_page_text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def create_empty_pdf_bytes() -> bytes:
    doc = fitz.open()
    doc.new_page()  # Page with zero text
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes



# ---------------- 1. NORMAL PDF TEST ----------------
def test_scenario_1_normal_pdf():
    pdf_bytes = create_normal_pdf_bytes("DocuMind features FastAPI, FAISS, and Sentence Transformers.")
    response = client.post(
        "/upload",
        files={"file": ("normal_doc.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "normal_doc.pdf"
    assert data["total_chunks"] >= 1
    assert data["embedding_dimension"] == 384


# ---------------- 2. SCANNED PDF TEST (OCR FALLBACK) ----------------
@patch("services.ocr_service.ocr_model")
def test_scenario_2_scanned_pdf(mock_ocr_model):
    mock_ocr_model.ocr.return_value = [[
        [[[1, 1], [2, 2], [3, 3], [4, 4]], ("Scanned PDF Extracted Text", 0.98)]
    ]]
    pdf_bytes = create_scanned_pdf_bytes()

    response = client.post(
        "/upload",
        files={"file": ("scanned_doc.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "scanned_doc.pdf"
    assert data["total_chunks"] >= 1


# ---------------- 3. IMAGE FILE TEST (.PNG OCR) ----------------
@patch("services.ocr_service.ocr_model")
def test_scenario_3_image_file(mock_ocr_model):
    mock_ocr_model.ocr.return_value = [[
        [[[1, 1], [2, 2], [3, 3], [4, 4]], ("Image Text Extraction Content", 0.95)]
    ]]
    img_bytes = create_png_image_bytes()

    response = client.post(
        "/upload",
        files={"file": ("sample_image.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample_image.png"
    assert data["total_chunks"] >= 1


# ---------------- 4. LARGE PDF TEST ----------------
def test_scenario_4_large_pdf():
    pdf_bytes = create_large_pdf_bytes(num_pages=8)
    response = client.post(
        "/upload",
        files={"file": ("large_doc.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "large_doc.pdf"
    assert data["total_chunks"] >= 5


# ---------------- 5. EMPTY PDF TEST ----------------
def test_scenario_5_empty_pdf():
    pdf_bytes = create_empty_pdf_bytes()
    response = client.post(
        "/upload",
        files={"file": ("empty_doc.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 422
    assert "Failed to extract text" in response.json()["detail"]


# ---------------- 6. WRONG FILE FORMAT TEST ----------------
def test_scenario_6_wrong_file():
    unsupported_bytes = b"Executable or binary content"
    response = client.post(
        "/upload",
        files={"file": ("malicious.exe", unsupported_bytes, "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


# ---------------- 7. MULTIPLE UPLOADS TEST ----------------
def test_scenario_7_multiple_uploads():
    pdf1 = create_normal_pdf_bytes("First document context about Python programming and web software development.")
    pdf2 = create_normal_pdf_bytes("Second document context about Machine Learning models and vector artificial intelligence.")

    res1 = client.post("/upload", files={"file": ("doc1.pdf", pdf1, "application/pdf")})
    assert res1.status_code == 200
    assert res1.json()["filename"] == "doc1.pdf"

    res2 = client.post("/upload", files={"file": ("doc2.pdf", pdf2, "application/pdf")})
    assert res2.status_code == 200
    assert res2.json()["filename"] == "doc2.pdf"

    # Query doc2 to confirm active document index updated cleanly
    rag_res = client.post("/rag", json={"query": "What is doc2 about?"})
    assert rag_res.status_code == 200



# ---------------- 8. WRONG QUESTION TEST (FALLBACK GUARANTEE) ----------------
def test_scenario_8_wrong_question():
    pdf = create_normal_pdf_bytes("DocuMind is an AI document assistant that processes PDFs.")
    client.post("/upload", files={"file": ("doc.pdf", pdf, "application/pdf")})

    response = client.post("/rag", json={"query": "What is the capital of Mars?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The document does not contain this information."


# ---------------- 9. CORRECT QUESTION TEST ----------------
@patch("utils.llm.get_qa_pipeline")
def test_scenario_9_correct_question(mock_get_pipeline):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"generated_text": "DocuMind uses FAISS for vector search."}]
    mock_get_pipeline.return_value = mock_pipeline

    pdf = create_normal_pdf_bytes("DocuMind uses FAISS for vector search and Sentence Transformers.")
    client.post("/upload", files={"file": ("doc.pdf", pdf, "application/pdf")})

    response = client.post("/rag", json={"query": "What does DocuMind use for vector search?"})
    assert response.status_code == 200
    data = response.json()
    assert "FAISS" in data["answer"]
    assert len(data["sources"]) >= 1
