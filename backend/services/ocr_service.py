import os
import io
import logging
from PIL import Image, ImageEnhance, ImageOps
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ocr_model = None
ocr_initialized = False

def init_ocr():
    """Lazy initializer for PaddleOCR model with safe fallback error handling."""
    global ocr_model, ocr_initialized
    if ocr_model is not None:
        return ocr_model
    if ocr_initialized:
        return ocr_model

    try:
        from paddleocr import PaddleOCR
        logger.info("[OCR-INIT] Initializing PaddleOCR engine...")
        try:
            ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
        except TypeError:
            ocr_model = PaddleOCR(lang='en')
        ocr_initialized = True
        logger.info("[OCR-INIT] PaddleOCR engine initialized successfully.")
    except Exception as e:
        logger.error(f"[OCR-INIT] Failed to initialize PaddleOCR engine: {e}")
        ocr_model = None
        ocr_initialized = True
    return ocr_model


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocesses images using Pillow to maximize OCR detection accuracy."""
    try:
        gray = image.convert('L')
        contrast = ImageEnhance.Contrast(gray).enhance(1.4)
        sharp = ImageEnhance.Sharpness(contrast).enhance(1.3)
        res = sharp.convert('RGB')
        res.shape = (res.height, res.width, 3)
        return res
    except Exception as e:
        logger.warning(f"[OCR-PREPROCESS] Preprocessing warning: {e}. Returning original RGB image.")
        res = image.convert('RGB')
        res.shape = (res.height, res.width, 3)
        return res


def extract_text_from_image_bytes(image_bytes: bytes, min_confidence: float = 0.50) -> str:
    """Extracts text from image bytes using PaddleOCR with confidence filtering."""
    engine = init_ocr()
    if engine is None:
        logger.error("[OCR-EXEC] PaddleOCR model is not initialized.")
        return ""

    temp_path = None
    try:
        if isinstance(image_bytes, bytes):
            img = Image.open(io.BytesIO(image_bytes))
        else:
            img = Image.open(image_bytes)

        prep_img = preprocess_image(img)
        
        temp_path = f"temp_ocr_{os.getpid()}_{id(image_bytes)}.png"
        prep_img.save(temp_path)

        logger.info(f"[OCR-EXEC] Running PaddleOCR line detection on image...")
        result = engine.ocr(temp_path, cls=True)

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        if not result or not result[0]:
            logger.info("[OCR-EXEC] No text detected in image by PaddleOCR.")
            return ""

        extracted_lines = []
        for line in result[0]:
            if len(line) >= 2 and len(line[1]) >= 2:
                text_content, confidence = line[1][0], line[1][1]
                if confidence >= min_confidence:
                    extracted_lines.append(text_content.strip())
                else:
                    logger.debug(f"[OCR-FILTER] Filtered out low confidence line ({confidence:.2f}): {text_content}")

        final_text = "\n".join(extracted_lines)
        logger.info(f"[OCR-EXEC] Successfully extracted {len(extracted_lines)} lines ({len(final_text)} chars).")
        return final_text

    except Exception as e:
        logger.error(f"[OCR-EXEC] Error during OCR processing: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return ""


class PageList(list):
    """List subclass that converts cleanly to a concatenated string when str() is called."""
    def __str__(self):
        return "\n\n".join([f"--- Page {p['page']} ---\n" + p["text"] for p in self])


def extract_text_with_ocr(pdf_path: str, min_confidence: float = 0.50) -> PageList:
    """Extracts text from scanned PDF by rendering pages to images in memory."""
    page_list = PageList()

    try:
        if pdf_path.endswith(('.png', '.jpg', '.jpeg')):
            with open(pdf_path, 'rb') as f:
                img_bytes = f.read()
            text = extract_text_from_image_bytes(img_bytes, min_confidence=min_confidence)
            if text:
                page_list.append({"page": 1, "text": text})
            return page_list

        doc = fitz.open(pdf_path)
        logger.info(f"[OCR-PDF] Processing {len(doc)} pages of scanned PDF: {pdf_path}")
        
        num_pages = len(doc)
        for page_num in range(num_pages):
            try:
                page = doc.load_page(page_num)
            except Exception:
                page = doc[page_num]

            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")

            text = extract_text_from_image_bytes(img_bytes, min_confidence=min_confidence)
            if text:
                page_list.append({"page": page_num + 1, "text": text})

        doc.close()
        logger.info(f"[OCR-PDF] Total text extracted across {len(page_list)} pages.")
        return page_list

    except Exception as e:
        logger.error(f"[OCR-PDF] Error extracting text from scanned PDF: {e}")
        return page_list


def clean_ocr_text(text: str) -> str:
    """Cleans extracted OCR text by removing unwanted control characters and normalizing whitespace."""
    if not text:
        return ""
    import re
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def should_use_ocr(raw_text: str, min_chars: int = 50) -> bool:
    """Determines whether OCR fallback is necessary based on native text length."""
    if not raw_text or len(raw_text.strip()) < min_chars:
        return True
    return False

