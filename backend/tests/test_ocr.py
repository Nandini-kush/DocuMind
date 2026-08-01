import sys
import os
import io
from PIL import Image
from unittest.mock import patch, MagicMock

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.ocr_service import (
    should_use_ocr,
    preprocess_image,
    clean_ocr_text,
    extract_text_with_ocr
)

def test_should_use_ocr_empty_text():
    assert should_use_ocr("") == True
    assert should_use_ocr("   ") == True

def test_should_use_ocr_short_text():
    assert should_use_ocr("Short text under 50 chars") == True

def test_should_use_ocr_long_text():
    long_text = "This is a much longer text that has more than 50 characters in it, so it should be considered valid extracted text."
    assert should_use_ocr(long_text) == False

def test_preprocess_image():
    # Test PIL image preprocessing
    img = Image.new("RGB", (100, 100), color="white")
    processed = preprocess_image(img)
    assert processed is not None
    assert processed.shape == (100, 100, 3)

def test_clean_ocr_text():
    raw = "  Hello \x00 World  \n "
    cleaned = clean_ocr_text(raw)
    assert cleaned == "Hello World"

@patch("services.ocr_service.fitz.open")
@patch("services.ocr_service.ocr_model")
def test_extract_text_from_pdf_with_ocr(mock_ocr_model, mock_fitz_open):
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_page = MagicMock()
    mock_pix = MagicMock()

    # Generate valid PNG bytes using PIL for the mock pixmap
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    valid_png_bytes = buf.getvalue()

    mock_pix.tobytes.return_value = valid_png_bytes
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.load_page.return_value = mock_page
    mock_fitz_open.return_value = mock_doc

    # Mock OCR output with confidence score 0.99
    mock_ocr_model.ocr.return_value = [[
        [[[1, 1], [2, 2], [3, 3], [4, 4]], ("Extracted text from page 1", 0.99)]
    ]]

    result = extract_text_with_ocr("test_document.pdf")
    
    assert len(result) == 1
    assert result[0]["page"] == 1
    assert result[0]["text"] == "Extracted text from page 1"

@patch("services.ocr_service.ocr_model")
def test_confidence_filtering(mock_ocr_model):
    # One line above threshold (0.95), one line below (0.20)
    mock_ocr_model.ocr.return_value = [[
        [[[1, 1], [2, 2], [3, 3], [4, 4]], ("Valid line", 0.95)],
        [[[1, 1], [2, 2], [3, 3], [4, 4]], ("Low confidence noise", 0.20)]
    ]]

    # Create temporary image file on disk
    temp_img_path = "test_image.png"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(temp_img_path)

    try:
        result = extract_text_with_ocr(temp_img_path)
        assert len(result) == 1
        assert result[0]["text"] == "Valid line"
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)


