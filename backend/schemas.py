from pydantic import BaseModel
from typing import Optional


class ProcessResponse(BaseModel):
    raw_text: str
    cleaned_text: str
    normalized_text: str
    word_count: int
    char_count: int
    processing_time_ms: float
    history_id: Optional[int] = None
    ocr_engine: str


class HistoryItem(BaseModel):
    id: int
    filename: str
    raw_text: str
    cleaned_text: str
    normalized_text: str
    word_count: int
    processing_time_ms: float
    created_at: str
