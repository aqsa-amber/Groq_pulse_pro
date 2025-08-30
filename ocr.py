import io, os
from typing import BinaryIO

# Extra imports for OCR helper
import pytesseract
from PIL import Image


def extract_text_from_file(uploaded_file: BinaryIO, filename: str = "", ocr_lang: str = "eng") -> str:
    """
    Attempts to extract text from common file types.
    Supported: TXT, DOCX, PDF, CSV, JSON, Images (OCR).
    """
    name = filename or getattr(uploaded_file, "name", "uploaded")
    ext = os.path.splitext(name)[1].lower()

    # Read raw bytes safely
    try:
        data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    except Exception:
        try:
            uploaded_file.seek(0)
            data = uploaded_file.read()
        except Exception:
            data = b""

    # TXT
    if ext == ".txt":
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return str(data)

    # DOCX
    if ext == ".docx":
        try:
            import docx
            from io import BytesIO
            doc = docx.Document(BytesIO(data))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            return "[Could not extract DOCX - missing 'python-docx' or corrupted file]"

    # PDF
    if ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(data))
            texts = []
            for p in reader.pages:
                try:
                    texts.append(p.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(texts)
        except Exception:
            return "[Could not extract PDF - missing 'PyPDF2' or unsupported PDF]"

    # CSV
    if ext == ".csv":
        try:
            import pandas as pd
            from io import StringIO
            return pd.read_csv(StringIO(data.decode("utf-8", errors="ignore"))).to_string()
        except Exception:
            return "[Could not extract CSV - missing 'pandas']"

    # JSON
    if ext == ".json":
        try:
            import json
            obj = json.loads(data.decode("utf-8", errors="ignore"))
            return json.dumps(obj, indent=2)
        except Exception:
            return "[Could not parse JSON file]"

    # Images (OCR)
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        try:
            im = Image.open(io.BytesIO(data))
            return pytesseract.image_to_string(im, lang=ocr_lang)
        except Exception:
            return "[Could not OCR image - missing 'pytesseract'/'Pillow' or tesseract not installed]"

    # Default fallback
    if data:
        return f"[Unsupported file type '{ext}']\nPreview: {str(data[:200])}..."
    return "[Empty file]"


# --- New helper for directly reading from a file path (simple OCR only) ---
def extract_text_from_image_file(file_path: str, ocr_lang: str = "eng") -> str:
    """
    Extracts text from an image file using pytesseract.
    """
    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img, lang=ocr_lang)
    except Exception:
        return "[Could not OCR image file - check path, Pillow, or tesseract installation]"
