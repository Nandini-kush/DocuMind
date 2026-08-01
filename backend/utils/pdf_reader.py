import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts native selectable text from a PDF file using PyMuPDF (fitz).
    Falls back to PyPDF2 if PyMuPDF encounters an unhandled exception.
    """
    text = ""

    # Primary: PyMuPDF (fitz) with safe context management
    try:
        import fitz
        with fitz.open(file_path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.warning(f"PyMuPDF native text extraction failed for {file_path}: {e}. Retrying with PyPDF2...")

    # Secondary Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PyPDF2 text extraction also failed for {file_path}: {e}")

    return text.strip()

