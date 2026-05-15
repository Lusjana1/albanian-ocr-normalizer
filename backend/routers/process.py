import os
import time
import uuid
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from database import delete_history_item, get_history, save_result
from schemas import HistoryItem, ProcessResponse
from services.cleaner import clean_text
from services.normalizer import normalize_text
from services.ocr_service import extract_text

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/process", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...)):
    # --- Validation ---
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG, PNG, WEBP, TIFF, or BMP image.",
        )

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20 MB.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Save to disk ---
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    file_id = str(uuid.uuid4()) + ext
    file_path = os.path.join(UPLOAD_DIR, file_id)
    with open(file_path, "wb") as f:
        f.write(content)

    start = time.perf_counter()
    try:
        # --- OCR ---
        raw_text, engine = extract_text(file_path)
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from the image.")

        # --- Clean ---
        cleaned_text = clean_text(raw_text)

        # --- Normalize ---
        normalized_text = normalize_text(cleaned_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        # Keep uploads for debugging; could add cleanup here
        pass

    elapsed_ms = (time.perf_counter() - start) * 1000
    word_count = len(cleaned_text.split())
    char_count = len(cleaned_text)

    history_id = await save_result(
        filename=file.filename or file_id,
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


@router.get("/history", response_model=list[HistoryItem])
async def get_processing_history(limit: int = 20):
    if limit < 1 or limit > 100:
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
