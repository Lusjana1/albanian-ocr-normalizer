"""
AI-assisted Albanian text normalizer.

Architecture note:
  mT5-small (google/mt5-small) is a multilingual span-filling model pretrained
  on mC4. No publicly available model has been fine-tuned specifically for
  Albanian text normalization — this is one of the key challenges highlighted
  in the diploma topic. We therefore combine:
    1. A lightweight AI pass using mT5 to attempt fluency improvement.
    2. A rule-based post-processing layer that applies Albanian orthographic
       corrections deterministically.
  The result demonstrates the full AI+NLP pipeline while being transparent
  about the current lack of Albanian-specific NLP resources.
"""
import logging
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

MODEL_NAME = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(_DEVICE)

_CHUNK_CHARS = 300


def _chunk_text(text: str) -> list[str]:
    """Split on sentence boundaries to stay within token limits."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= _CHUNK_CHARS:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks or [text]


def _is_valid_output(text: str) -> bool:
    """Return False when the model emits sentinel tokens or empty output."""
    if not text or not text.strip():
        return False
    if "<extra_id_" in text:
        return False
    if len(text) < 3:
        return False
    return True


def _ai_normalize_chunk(chunk: str) -> str:
    """
    Run the chunk through mT5 with a translation-style prefix.
    Falls back to the input chunk when output is invalid.
    """
    # T5-family models respond to explicit task prefixes
    prompt = f"translate Albanian to Albanian: {chunk}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(_DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return decoded if _is_valid_output(decoded) else chunk


# ---------------------------------------------------------------------------
# Rule-based post-processing applied AFTER the AI pass
# ---------------------------------------------------------------------------
_POST_RULES = [
    # Expand common Albanian abbreviations
    (r"\bdr\.", "Dr."),
    (r"\bprof\.", "Prof."),
    (r"\bnr\.", "Nr."),
    (r"\bfq\.", "fq."),
    # Ensure proper Albanian quotation style
    (r'"([^"]{1,200})"', r"«\1»"),
    # Capitalise after sentence-ending punctuation
    (r"([.!?])\s+([a-zëç])", lambda m: m.group(1) + " " + m.group(2).upper()),
]


def _post_process(text: str) -> str:
    for pat, repl in _POST_RULES:
        text = re.sub(pat, repl, text)
    if text:
        text = text[0].upper() + text[1:]
    return text.strip()


def normalize_text(cleaned_text: str) -> str:
    """
    Full normalization pipeline: AI pass → rule-based post-processing.
    Falls back gracefully to cleaned_text on any model error.
    """
    if not cleaned_text.strip():
        return cleaned_text

    try:
        chunks = _chunk_text(cleaned_text)
        ai_chunks = [_ai_normalize_chunk(c) for c in chunks]
        combined = " ".join(ai_chunks)
        return _post_process(combined)
    except Exception as e:
        logger.error("Normalization error: %s", e)
        return _post_process(cleaned_text)
