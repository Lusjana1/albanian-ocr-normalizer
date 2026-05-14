from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os

from ocr import extract_text
from normalizer import normalize_text

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(UPLOAD_DIR, file_id)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # OCR step
    raw_text = extract_text(file_path)

    # Normalization step
    normalized_text = normalize_text(raw_text)

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text
    }