"""
OCR service — multi-variant preprocessing + confidence-scored engine fallback.

Pipeline:
  1. Load image → upscale → deskew
  2. Generate 3 preprocessed variants (adaptive, Otsu, sharpened-adaptive)
  3. Run Tesseract on each variant; pick winner by mean word confidence
  4. If best Tesseract confidence is low, also try extra PSM modes
  5. Fall back to PaddleOCR, then EasyOCR if Tesseract fails or produces nothing
"""
import logging
import shutil
import tempfile
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None

_paddle_ocr = None
_easy_ocr = None


# ---------------------------------------------------------------------------
# Lazy singletons for heavy engines
# ---------------------------------------------------------------------------

def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(lang="sq", show_log=False)
    return _paddle_ocr


def _get_easyocr():
    global _easy_ocr
    if _easy_ocr is None:
        import easyocr
        # Albanian uses the Latin character set; 'en' model covers it well
        _easy_ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easy_ocr


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_image(image_path: str) -> np.ndarray:
    """Load image via OpenCV, with PIL fallback for exotic formats."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _save_temp_png(arr: np.ndarray) -> str:
    """Write a numpy array to a temp PNG and return the path."""
    import cv2
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, arr)
    return path


# ---------------------------------------------------------------------------
# Preprocessing building blocks
# ---------------------------------------------------------------------------

def _upscale(img: np.ndarray, min_w: int = 1400, min_h: int = 300) -> np.ndarray:
    """Scale up small images so OCR engines see adequately-sized glyphs."""
    import cv2
    h, w = img.shape[:2]
    if w >= min_w and h >= min_h:
        return img
    scale = max(min_w / w, min_h / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


def _estimate_skew(gray: np.ndarray) -> float:
    """
    Estimate document skew angle via projection-profile sweep.
    Works on a downscaled thumbnail so the 61-iteration loop stays fast
    regardless of input resolution.
    """
    import cv2

    # Downscale to max 600 px wide for speed — angle accuracy is unaffected
    h, w = gray.shape[:2]
    if w > 600:
        scale = 600 / w
        thumb = cv2.resize(gray, (600, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        thumb = gray

    _, thresh = cv2.threshold(thumb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best_angle, best_score = 0.0, -1.0
    cx, cy = thresh.shape[1] // 2, thresh.shape[0] // 2
    # 1-degree steps are sufficient for document skew correction
    for angle in np.arange(-12, 12.5, 1.0):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(thresh, M, (thresh.shape[1], thresh.shape[0]),
                                  flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(np.sum(rotated, axis=1).astype(float)))
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def _deskew(img: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Rotate image to neutralise detected skew."""
    import cv2
    try:
        angle = _estimate_skew(gray)
    except Exception:
        return img
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    corrected = cv2.warpAffine(img, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    logger.debug("Deskewed %.1f°", angle)
    return corrected


def _add_border(img: np.ndarray, size: int = 30, color: int = 255) -> np.ndarray:
    """Add a white border so text at edges is not clipped by the OCR engine."""
    import cv2
    return cv2.copyMakeBorder(img, size, size, size, size,
                               cv2.BORDER_CONSTANT, value=color)


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------

def _build_variants(image_path: str) -> list[tuple[Image.Image, str]]:
    """
    Return a list of (PIL_image, variant_label) ready to feed to Tesseract.

    Variants produced:
      A — adaptive threshold   → best for uneven lighting, shadows, yellowed paper
      B — Otsu threshold       → best for clean, high-contrast scans
      C — sharpened adaptive   → best for soft/blurry text
      D — heavy-denoise + Otsu → best for WhatsApp-compressed or grain-heavy images
    """
    import cv2

    img = _load_image(image_path)
    img = _upscale(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = _deskew(img, gray)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add a border before any variant so edge glyphs are safe
    gray = _add_border(gray)

    # ---- shared base operations ----
    # Light denoise (preserves sharp edges better than heavy denoising)
    base = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE — adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(base)

    # ---- Variant A: adaptive threshold ----
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=10,
    )
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, open_k)

    # ---- Variant B: Otsu threshold ----
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- Variant C: sharpened + tighter adaptive ----
    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_k)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    sharp_adaptive = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=15, C=8,
    )

    # ---- Variant D: heavy denoise + Otsu (for noisy/compressed images) ----
    heavy = cv2.fastNlMeansDenoising(gray, h=20, templateWindowSize=7, searchWindowSize=21)
    heavy_clahe = clahe.apply(heavy)
    _, heavy_otsu = cv2.threshold(heavy_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [
        (Image.fromarray(adaptive),    "adaptive"),
        (Image.fromarray(otsu),        "otsu"),
        (Image.fromarray(sharp_adaptive), "sharp+adaptive"),
        (Image.fromarray(heavy_otsu),  "heavy-denoise+otsu"),
    ]


def _pil_fallback_variants(image_path: str) -> list[tuple[Image.Image, str]]:
    """PIL-only fallback when OpenCV is unavailable."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w < 1400:
        scale = 1400 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = img.convert("L")
    sharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
    contrast = ImageEnhance.Contrast(img).enhance(2.0)
    return [(sharp, "pil-sharp"), (contrast, "pil-contrast")]


# ---------------------------------------------------------------------------
# Tesseract — confidence-scored multi-variant selection
# ---------------------------------------------------------------------------

# Primary config — Albanian LSTM with automatic page layout analysis
_PRIMARY_LANG = "sqi+eng"
_PRIMARY_CFG  = "--oem 3 --psm 6"

# Extra configs tried only when the primary confidence is below threshold
_EXTRA_CONFIGS = [
    ("sqi+eng", "--oem 3 --psm 3"),   # fully automatic — good for mixed layouts
    ("sqi+eng", "--oem 3 --psm 11"),  # sparse text — maximises character recall
    ("sqi+eng", "--oem 3 --psm 4"),   # single column — for scanned books
    ("sqi",     "--oem 3 --psm 6"),   # Albanian-only (no English interference)
    ("eng",     "--oem 3 --psm 6"),   # English fallback for Latin OCR
]

_LOW_CONFIDENCE_THRESHOLD = 45  # mean word confidence below this → try extra configs


def _run_tesseract(pil_img: Image.Image, lang: str, config: str) -> tuple[str, float]:
    """
    Run Tesseract and return (text, mean_word_confidence).
    Uses image_to_data for objective confidence, image_to_string for clean text.
    """
    import pytesseract

    try:
        data = pytesseract.image_to_data(
            pil_img, lang=lang, config=config,
            output_type=pytesseract.Output.DICT,
        )
        confs = [
            int(c)
            for c, t in zip(data["conf"], data["text"])
            if int(c) > 0 and str(t).strip()
        ]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
    except Exception:
        mean_conf = 0.0

    try:
        text = pytesseract.image_to_string(pil_img, lang=lang, config=config).strip()
    except Exception:
        text = ""

    return text, mean_conf


def _extract_with_tesseract(image_path: str) -> tuple[str, float]:
    """
    Try every preprocessing variant with the primary Tesseract config.
    Pick the variant with the highest mean confidence.
    If that confidence is still below the threshold, try extra PSM configs too.
    Returns (best_text, best_mean_confidence).
    """
    try:
        variants = _build_variants(image_path)
    except Exception as e:
        logger.warning("OpenCV variant building failed, using PIL fallback: %s", e)
        variants = _pil_fallback_variants(image_path)

    best_text, best_conf = "", 0.0

    # --- Phase 1: primary config on all variants ---
    for pil_img, label in variants:
        text, conf = _run_tesseract(pil_img, _PRIMARY_LANG, _PRIMARY_CFG)
        logger.debug("Tesseract variant=%s conf=%.1f chars=%d", label, conf, len(text))
        if conf > best_conf or (conf == best_conf and len(text) > len(best_text)):
            best_conf, best_text = conf, text

    # --- Phase 2: if confidence is still low, try extra configs on best variant ---
    if best_conf < _LOW_CONFIDENCE_THRESHOLD:
        best_variant = variants[0][0]  # default to first if no clear winner
        for pil_img, label in variants:
            _, conf = _run_tesseract(pil_img, _PRIMARY_LANG, _PRIMARY_CFG)
            if conf == best_conf:
                best_variant = pil_img
                break

        for lang, cfg in _EXTRA_CONFIGS:
            text, conf = _run_tesseract(best_variant, lang, cfg)
            logger.debug("Tesseract extra lang=%s conf=%.1f chars=%d", lang, conf, len(text))
            if conf > best_conf or (conf == best_conf and len(text) > len(best_text)):
                best_conf, best_text = conf, text
            if best_conf >= _LOW_CONFIDENCE_THRESHOLD:
                break

    return best_text, best_conf


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

def _parse_paddle_new_api(results) -> list[tuple[str, float]]:
    """Parse PaddleOCR v3 predict() result → [(text, confidence)]."""
    items = []
    for res in results:
        if not hasattr(res, "json"):
            continue
        try:
            data = res.json.get("res", {})
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [1.0] * len(texts))
            for t, s in zip(texts, scores):
                if t and str(t).strip():
                    items.append((str(t).strip(), float(s)))
        except Exception:
            pass
    return items


def _parse_paddle_old_api(results) -> list[tuple[str, float]]:
    """Parse PaddleOCR v2 ocr() result → [(text, confidence)]."""
    items = []
    flat = results[0] if results and isinstance(results[0], list) else results
    for item in (flat or []):
        if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            payload = item[1]
            text = payload[0] if isinstance(payload, (list, tuple)) else str(payload)
            conf = float(payload[1]) if isinstance(payload, (list, tuple)) and len(payload) > 1 else 1.0
            if text and str(text).strip():
                items.append((str(text).strip(), conf))
        except Exception:
            pass
    return items


def _extract_with_paddle(image_path: str) -> tuple[str, float]:
    """Returns (text, mean_confidence)."""
    ocr = _get_paddle()
    items: list[tuple[str, float]] = []

    try:
        results = ocr.predict(image_path)
        items = _parse_paddle_new_api(results)
    except AttributeError:
        pass
    except Exception as e:
        logger.warning("PaddleOCR predict() error: %s", e)

    if not items:
        try:
            results = ocr.ocr(image_path, cls=True)
            items = _parse_paddle_old_api(results)
        except Exception as e:
            logger.warning("PaddleOCR ocr() error: %s", e)

    if not items:
        return "", 0.0

    text = "\n".join(t for t, _ in items)
    mean_conf = sum(s for _, s in items) / len(items) * 100  # normalise to 0-100 scale
    return text, mean_conf


# ---------------------------------------------------------------------------
# EasyOCR — with confidence filtering
# ---------------------------------------------------------------------------

_EASYOCR_MIN_CONF = 0.35  # discard words below 35 % confidence


def _extract_with_easyocr(image_path: str) -> tuple[str, float]:
    """Returns (text, mean_confidence_pct)."""
    reader = _get_easyocr()
    try:
        raw = reader.readtext(image_path, detail=1, paragraph=False)
        # raw is a list of (bbox, text, confidence)
        accepted = [(text, conf) for _, text, conf in raw
                    if conf >= _EASYOCR_MIN_CONF and text.strip()]
        if not accepted:
            return "", 0.0
        text = "\n".join(t for t, _ in accepted)
        mean_conf = sum(c for _, c in accepted) / len(accepted) * 100
        return text, mean_conf
    except Exception as e:
        logger.warning("EasyOCR failed: %s", e)
        return "", 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(image_path: str) -> tuple[str, str]:
    """
    Run multi-engine, confidence-scored OCR.
    Returns (best_text, engine_name_used).
    """
    results: list[tuple[str, float, str]] = []  # (text, confidence, engine)

    # --- Tesseract ---
    if TESSERACT_AVAILABLE:
        try:
            text, conf = _extract_with_tesseract(image_path)
            if text:
                results.append((text, conf, "Tesseract (sqi+eng)"))
                logger.info("Tesseract: %d chars, confidence %.1f", len(text), conf)
        except Exception as e:
            logger.warning("Tesseract pipeline failed: %s", e)

    # --- PaddleOCR ---
    try:
        text, conf = _extract_with_paddle(image_path)
        if text:
            results.append((text, conf, "PaddleOCR (sq)"))
            logger.info("PaddleOCR: %d chars, confidence %.1f", len(text), conf)
    except Exception as e:
        logger.warning("PaddleOCR pipeline failed: %s", e)

    # --- EasyOCR ---
    try:
        text, conf = _extract_with_easyocr(image_path)
        if text:
            results.append((text, conf, "EasyOCR (en)"))
            logger.info("EasyOCR: %d chars, confidence %.1f", len(text), conf)
    except Exception as e:
        logger.warning("EasyOCR pipeline failed: %s", e)

    if not results:
        return "", "none"

    # Pick the engine with the highest confidence
    best_text, best_conf, best_engine = max(results, key=lambda x: x[1])

    if len(results) > 1:
        logger.info(
            "Engine selected: %s (conf=%.1f) over %s",
            best_engine,
            best_conf,
            [e for _, _, e in results if e != best_engine],
        )

    return best_text, best_engine
