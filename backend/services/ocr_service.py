"""
OCR service: Tesseract (primary) with PaddleOCR fallback.
Tesseract uses Albanian tessdata (sqi); PaddleOCR uses the Latin/sq model.
"""
import logging
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None

# Lazy singleton for PaddleOCR (expensive to init)
_paddle_ocr = None


def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(lang="sq")
    return _paddle_ocr


def _preprocess(image_path: str) -> Image.Image:
    """Light preprocessing to improve OCR accuracy."""
    img = Image.open(image_path).convert("RGB")
    # Upscale small images
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Sharpen and increase contrast slightly
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    return img


def _extract_with_tesseract(image_path: str) -> str:
    import pytesseract

    img = _preprocess(image_path)
    # Try Albanian first, fall back to English
    for lang in ("sqi+eng", "eng"):
        try:
            text = pytesseract.image_to_string(
                img,
                lang=lang,
                config="--oem 3 --psm 6",
            )
            text = text.strip()
            if text:
                return text
        except Exception as e:
            logger.warning("Tesseract lang=%s failed: %s", lang, e)
    return ""


def _extract_with_paddle(image_path: str) -> str:
    ocr = _get_paddle()
    results = ocr.predict(image_path)
    lines = []
    for res in results:
        rec_texts = res.json.get("res", {}).get("rec_texts", [])
        lines.extend([t for t in rec_texts if t and t.strip()])
    return "\n".join(lines)


def extract_text(image_path: str) -> tuple[str, str]:
    """
    Returns (text, engine_name).
    Tries Tesseract first, falls back to PaddleOCR.
    """
    if TESSERACT_AVAILABLE:
        try:
            text = _extract_with_tesseract(image_path)
            if text:
                return text, "Tesseract (sqi)"
        except Exception as e:
            logger.warning("Tesseract extraction failed: %s", e)

    try:
        text = _extract_with_paddle(image_path)
        return text, "PaddleOCR (sq)"
    except Exception as e:
        logger.error("PaddleOCR extraction failed: %s", e)
        return "", "none"
