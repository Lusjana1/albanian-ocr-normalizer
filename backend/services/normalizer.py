"""
Albanian text normalizer — mT5-small AI pass + rule-based post-processing.

Performance notes
-----------------
* num_beams=1  (greedy decoding) — 4× faster than beam search; quality is
  nearly identical for short normalization tasks on a pre-trained mT5.
* max_new_tokens=100 — enough for a 300-char chunk; prevents runaway generation.
* Texts longer than _AI_CHAR_LIMIT skip the AI pass entirely and go straight
  to rule-based post-processing.  This keeps the endpoint responsive when OCR
  extracts a long document: rules run in milliseconds, the AI adds seconds per
  chunk and would easily exceed the caller's 90-second timeout.
"""
import logging
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(_DEVICE)

# Skip the AI pass for texts longer than this — too slow on CPU
_AI_CHAR_LIMIT  = 1200
_CHUNK_CHARS    = 280   # keep well below the 300-token model limit
_MAX_NEW_TOKENS = 100   # was 256 — prevents runaway generation


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk(text: str) -> list[str]:
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


# ---------------------------------------------------------------------------
# Single-chunk AI inference
# ---------------------------------------------------------------------------

def _ai_normalize_chunk(chunk: str) -> str:
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
            max_new_tokens=_MAX_NEW_TOKENS,
            num_beams=1,          # greedy — was 4, 4× faster
            do_sample=False,
            early_stopping=False, # irrelevant for greedy but explicit
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Reject outputs that are clearly model artifacts
    if not decoded or "<extra_id_" in decoded or len(decoded) < 3:
        return chunk
    return decoded


# ---------------------------------------------------------------------------
# Rule-based post-processing (runs even when AI is skipped)
# ---------------------------------------------------------------------------

_POST_RULES = [
    (r"\bdr\.",   "Dr."),
    (r"\bprof\.", "Prof."),
    (r"\bnr\.",   "Nr."),
    (r"\bfq\.",   "fq."),
    (r'"([^"\n]{1,200})"', r"«\1»"),
    (r"([.!?])\s+([a-zëç])", lambda m: m.group(1) + " " + m.group(2).upper()),
]


def _post_process(text: str) -> str:
    for pat, repl in _POST_RULES:
        text = re.sub(pat, repl, text)
    if text:
        text = text[0].upper() + text[1:]
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_text(cleaned_text: str) -> str:
    """
    Normalize Albanian OCR text.

    Short texts  (<= _AI_CHAR_LIMIT): AI pass then rule-based post-processing.
    Long texts   (>  _AI_CHAR_LIMIT): rule-based post-processing only.

    Any exception inside the AI pass falls back to the input text so the
    caller always receives something useful.
    """
    if not cleaned_text.strip():
        return cleaned_text

    if len(cleaned_text) > _AI_CHAR_LIMIT:
        logger.info(
            "Text too long for AI normalization (%d chars > %d limit) — using rules only",
            len(cleaned_text), _AI_CHAR_LIMIT,
        )
        return _post_process(cleaned_text)

    try:
        chunks    = _chunk(cleaned_text)
        logger.debug("Normalizing %d chunk(s)", len(chunks))
        ai_chunks = []
        for i, c in enumerate(chunks):
            result = _ai_normalize_chunk(c)
            logger.debug("Chunk %d/%d done (%d→%d chars)", i + 1, len(chunks), len(c), len(result))
            ai_chunks.append(result)
        return _post_process(" ".join(ai_chunks))
    except Exception as e:
        logger.error("AI normalization failed: %s — returning cleaned text", e)
        return _post_process(cleaned_text)
