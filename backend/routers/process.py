import asyncio
import io
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from database import delete_history_item, get_history, save_result
from schemas import HistoryItem, ProcessResponse
from services.cleaner import clean_text
from services.normalizer import normalize_text
from services.ocr_service import extract_text

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_BYTES = 30 * 1024 * 1024  # 30 MB

# Single shared thread-pool for all blocking CPU work.
# max_workers=2 prevents OOM when multiple images are uploaded concurrently.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr-worker")

# Per-stage timeouts (seconds)
_TIMEOUT_OCR       = 180  # Tesseract + PaddleOCR + EasyOCR with preprocessing
_TIMEOUT_NORMALIZE =  90  # mT5 inference (greedy decoding on CPU)


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

_REJECTED_MIME_PREFIXES = ("application/pdf", "video/", "audio/")


def _validate_image(content: bytes, filename: str) -> None:
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=415,
            detail=(
                f"'{filename}' is not a recognised image format. "
                "Supported: JPEG, PNG, WebP, TIFF, BMP. "
                "Make sure the file is not corrupted."
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image appears damaged or incomplete: {e}. Try re-exporting it.",
        )


# ---------------------------------------------------------------------------
# Async helpers — offload blocking calls so the event loop stays free
# ---------------------------------------------------------------------------

async def _run_blocking(fn, *args, timeout: float):
    """
    Run a synchronous function in the thread-pool executor.
    Raises HTTPException(504) if it exceeds `timeout` seconds.
    """
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, fn, *args),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise  # re-raise so callers can decide how to handle it


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/process", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...)):
    filename = file.filename or "image"
    content_type = (file.content_type or "").lower()

    for bad in _REJECTED_MIME_PREFIXES:
        if content_type.startswith(bad):
            raise HTTPException(
                status_code=415,
                detail=f"'{filename}' is a {content_type} — only images are accepted.",
            )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail=f"'{filename}' is empty.")

    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(content) // (1024*1024):.0f} MB — max is {MAX_FILE_BYTES // (1024*1024)} MB.",
        )

    _validate_image(content, filename)

    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp", ".gif"}:
        ext = ".jpg"
    file_path = os.path.join(UPLOAD_DIR, str(uuid.uuid4()) + ext)
    with open(file_path, "wb") as f:
        f.write(content)

    total_start = time.perf_counter()

    # ------------------------------------------------------------------ OCR
    logger.info("[%s] Stage 1/3: OCR started", filename)
    t = time.perf_counter()
    try:
        raw_text, engine = await _run_blocking(extract_text, file_path, timeout=_TIMEOUT_OCR)
    except asyncio.TimeoutError:
        logger.error("[%s] OCR timed out after %ds", filename, _TIMEOUT_OCR)
        raise HTTPException(
            status_code=504,
            detail=(
                f"OCR timed out after {_TIMEOUT_OCR} seconds. "
                "The image may be too large or complex. "
                "Try a smaller image or one with higher contrast."
            ),
        )
    except Exception as e:
        logger.exception("[%s] OCR crashed", filename)
        raise HTTPException(status_code=500, detail=f"OCR error: {e}")

    logger.info("[%s] OCR done in %.1fs — engine=%s chars=%d",
                filename, time.perf_counter() - t, engine, len(raw_text))

    if not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted. The image may contain only graphics, "
                "be too low-resolution, or the text may be too faint. "
                "Try scanning at 300 DPI or higher."
            ),
        )

    # --------------------------------------------------------------- CLEAN
    logger.info("[%s] Stage 2/3: Cleaning (%d chars)", filename, len(raw_text))
    t = time.perf_counter()
    try:
        # clean_text is pure regex — fast, but run in executor anyway to be safe
        cleaned_text = await _run_blocking(clean_text, raw_text, timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("[%s] Cleaner timed out — using raw text", filename)
        cleaned_text = raw_text
    except Exception as e:
        logger.warning("[%s] Cleaner failed (%s) — using raw text", filename, e)
        cleaned_text = raw_text

    logger.info("[%s] Clean done in %.1fs — %d chars", filename, time.perf_counter() - t, len(cleaned_text))

    # ----------------------------------------------------------- NORMALIZE
    logger.info("[%s] Stage 3/3: Normalizing (%d chars)", filename, len(cleaned_text))
    t = time.perf_counter()
    try:
        normalized_text = await _run_blocking(normalize_text, cleaned_text, timeout=_TIMEOUT_NORMALIZE)
    except asyncio.TimeoutError:
        logger.warning("[%s] Normalization timed out after %ds — using cleaned text",
                       filename, _TIMEOUT_NORMALIZE)
        normalized_text = cleaned_text   # graceful fallback
    except Exception as e:
        logger.warning("[%s] Normalization failed (%s) — using cleaned text", filename, e)
        normalized_text = cleaned_text   # graceful fallback

    logger.info("[%s] Normalize done in %.1fs", filename, time.perf_counter() - t)

    # --------------------------------------------------------- SAVE + RETURN
    elapsed_ms = (time.perf_counter() - total_start) * 1000
    word_count = len(cleaned_text.split())
    char_count  = len(cleaned_text)

    logger.info("[%s] Total pipeline: %.1fs", filename, elapsed_ms / 1000)

    history_id = await save_result(
        filename=filename,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        normalized_text=normalized_text,
        word_count=word_count,
        processing_time_ms=round(elapsed_ms, 1),
    )

    return ProcessResponse(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        normalized_text=normalized_text,
        word_count=word_count,
        char_count=char_count,
        processing_time_ms=round(elapsed_ms, 1),
        history_id=history_id,
        ocr_engine=engine,
    )


# ---------------------------------------------------------------------------
# History routes
# ---------------------------------------------------------------------------

@router.get("/history", response_model=list[HistoryItem])
async def get_processing_history(limit: int = 20):
    if not (1 <= limit <= 100):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")
    rows = await get_history(limit)
    return [
        HistoryItem(
            id=r["id"],
            filename=r["filename"],
            raw_text=r["raw_text"],
            cleaned_text=r["cleaned_text"],
            normalized_text=r["normalized_text"],
            word_count=r["word_count"],
            processing_time_ms=r["processing_time_ms"],
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]


@router.delete("/history/{item_id}")
async def delete_history(item_id: int):
    await delete_history_item(item_id)
    return JSONResponse({"deleted": item_id})
