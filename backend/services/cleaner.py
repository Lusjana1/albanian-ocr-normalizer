"""
Rule-based Albanian text cleaner for OCR output.

Pipeline order:
  1. Hyphenated line-break joining   (e.g. "fja-\nlë" → "fjalë")
  2. OCR character substitutions     (digit/glyph confusions)
  3. Albanian character recovery     (e→ë, c→ç in known words)
  4. Noise removal
  5. Whitespace normalisation
  6. Punctuation fixes
  7. Sentence capitalisation
"""
import re


# ---------------------------------------------------------------------------
# Step 1 — Hyphenated line-break joining
# ---------------------------------------------------------------------------
# Books, poems, and scanned documents often hyphenate words at line ends.
# This must run before any other substitution so hyphens are still intact.

def _join_hyphenated_breaks(text: str) -> str:
    # "fjal-\nlë" → "fjalë"   (word continues on next line without space)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # "fjal-  \n  lë" — already covered by the \s* above
    return text


# ---------------------------------------------------------------------------
# Step 2 — OCR character substitutions  (digit / glyph confusions)
# ---------------------------------------------------------------------------
# REMOVED:  rn → m   (breaks "alternativë", "qeverni", "shërbim", etc.)
# REMOVED:  li → h   (breaks "politik", "familje", "libertar", etc.)

_CHAR_SUBS = [
    # Digits mistaken for letters inside words
    (r"(?<=[A-Za-zëçËÇ])0(?=[A-Za-zëçËÇ])", "o"),
    (r"(?<=[A-Za-zëçËÇ])1(?=[A-Za-zëçËÇ])", "i"),
    (r"\b1(?=[A-Za-zëçËÇ])", "I"),
    # Pipe / broken-bar used as l or i
    (r"\|(?=[a-z])", "l"),
    (r"(?<=[a-z])\|", "l"),
    # Backtick / grave used as apostrophe
    (r"`", "'"),
    # Tilde used as dash in some old scans
    (r"~", "-"),
]


# ---------------------------------------------------------------------------
# Step 3 — Albanian character recovery
# ---------------------------------------------------------------------------
# OCR frequently substitutes ë with é/è/ê/ä/ö and ç with Ć/ć/c-cedilla.
# Additionally, a small word-level dictionary corrects the most common cases
# where plain 'e' is output instead of 'ë' and plain 'c' instead of 'ç'.

_DIACRITIC_FIXES = [
    # Accented Latin variants → Albanian ë
    (r"[éèêëěĕ]", "ë"),
    (r"[ÉÈÊËĚ]",  "Ë"),
    # Accented variants → Albanian ç
    (r"[ćčĉ]",    "ç"),
    (r"[ĆČĈ]",    "Ç"),
    # German-style umlauts that creep in via wrong codepage detection
    (r"ä(?=[a-zëç])", "ë"),   # ä mid-word → ë  (conservative)
]

# Albanian digraph reconstruction — OCR often splits digraphs with spaces/hyphens.
_DIGRAPH_FIXES = [
    (r"(?i)\b(d)\s*-?\s*(h)\b", r"\1\2"),
    (r"(?i)\b(g)\s*-?\s*(j)\b", r"\1\2"),
    (r"(?i)\b(n)\s*-?\s*(j)\b", r"\1\2"),
    (r"(?i)\b(s)\s*-?\s*(h)\b", r"\1\2"),
    (r"(?i)\b(t)\s*-?\s*(h)\b", r"\1\2"),
    (r"(?i)\b(x)\s*-?\s*(h)\b", r"\1\2"),
    (r"(?i)\b(z)\s*-?\s*(h)\b", r"\1\2"),
]

# Word-level corrections: only apply when the exact OCR form is unambiguous.
# Each tuple is (pattern_to_match, replacement).
# Rules are intentionally conservative — no 'ne'→'në' because 'ne' is valid.
_ALBANIAN_WORD_FIXES = [
    # çdo (every) — 'cdo' is never a valid Albanian word
    (r"\bcdo\b",    "çdo"),
    (r"\bCdo\b",    "Çdo"),
    (r"\bCDO\b",    "ÇDO"),
    # çfarë (what/which) — various OCR forms
    (r"\bcfare\b",  "çfarë"),
    (r"\bcfar\b",   "çfar"),
    # është (is) — 'eshte' unambiguous
    (r"\beshte\b",  "është"),
    (r"\bEshte\b",  "Është"),
    # janë (are) — 'jane' unambiguous in Albanian context
    (r"\bjane\b",   "janë"),
    # kanë (have 3rd pl.) — 'kane' unambiguous
    (r"\bkane\b",   "kanë"),
    # bënë (they did) — 'bene' has meaning in other languages but rare here
    (r"\bbene\b",   "bënë"),
    # gjithë (all) — 'gjithe' unambiguous
    (r"\bgjithe\b", "gjithë"),
    # për (for) — 'per' is a safe fix in Albanian-dominant text
    (r"\bper\b",    "për"),
    (r"\bPer\b",    "Për"),
    # që (that/which) — 'qe' unambiguous as Albanian connector
    (r"\bqe\b",     "që"),
    (r"\bQe\b",     "Që"),
    # më (me/more) — 'me' is valid as a standalone word, skip
    # të (to/the) — 'te' is also valid, skip
    # në (in) — 'ne' ambiguous, skip
    # si (like/how) — 'si' is correct already
]


# ---------------------------------------------------------------------------
# Step 4 — Noise removal
# ---------------------------------------------------------------------------

_NOISE_PATTERNS = [
    # Long runs of the same non-alphanumeric character (e.g. ".....", "-----")
    (r"([^A-Za-z0-9ëçËÇ\s])\1{3,}", ""),
    # Lines that contain only punctuation / symbols (separator lines in scans)
    (r"(?m)^[^A-Za-z0-9ëçËÇ]+$", ""),
    # Standalone page numbers on their own line (1–4 digits)
    (r"(?m)^\s*\d{1,4}\s*$", ""),
    # Zero-width and control characters (except \t and \n)
    (r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", ""),
    # Isolated single non-letter, non-digit characters surrounded by spaces
    # (common OCR artifact: stray dots, pipes, underscores between words)
    (r"(?<= )[^A-Za-z0-9ëçËÇ'\"«»\-](?= )", ""),
]


# ---------------------------------------------------------------------------
# Step 5 — Whitespace normalisation
# ---------------------------------------------------------------------------

def _normalise_whitespace(text: str) -> str:
    # Collapse multiple spaces / tabs on a single line
    text = re.sub(r"[ \t]+", " ", text)
    # Remove trailing spaces before newlines
    text = re.sub(r" +\n", "\n", text)
    # Collapse 3+ blank lines to a single blank line (preserves paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace from the whole text
    return text.strip()


# ---------------------------------------------------------------------------
# Step 6 — Punctuation fixes
# ---------------------------------------------------------------------------

_PUNCT_PATTERNS = [
    # Remove space before punctuation
    (r"\s+([,.!?;:»\)])", r"\1"),
    # Ensure space after sentence-ending punctuation before a capital letter
    (r"([.!?])([A-ZËÇА-Я])", r"\1 \2"),
    # Collapse repeated punctuation (e.g. ".." → ".")
    (r"([.!?]){2,}", r"\1"),
    # Straight double quotes → Albanian guillemets
    (r'"([^"\n]{1,300})"', r"«\1»"),
]


# ---------------------------------------------------------------------------
# Step 7 — Sentence capitalisation
# ---------------------------------------------------------------------------

def _capitalise_sentences(text: str) -> str:
    # Capitalise the very first character
    if text:
        text = text[0].upper() + text[1:]
    # Capitalise first letter after sentence-ending punctuation + whitespace
    text = re.sub(
        r"([.!?]\s+)([a-zëç])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _apply(text: str, patterns: list) -> str:
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    return text


def clean_text(raw: str) -> str:
    """Full cleaning pipeline for Albanian OCR output."""
    text = raw

    # 1. Hyphenated line-break joining (must be first)
    text = _join_hyphenated_breaks(text)

    # 2. OCR character substitutions
    text = _apply(text, _CHAR_SUBS)

    # 3a. Diacritic / character recovery
    text = _apply(text, _DIACRITIC_FIXES)

    # 3b. Albanian digraph reconstruction
    text = _apply(text, _DIGRAPH_FIXES)

    # 3c. Word-level Albanian corrections
    text = _apply(text, _ALBANIAN_WORD_FIXES)

    # 4. Noise removal
    text = _apply(text, _NOISE_PATTERNS)

    # 5. Whitespace normalisation
    text = _normalise_whitespace(text)

    # 6. Punctuation fixes
    text = _apply(text, _PUNCT_PATTERNS)

    # 7. Sentence capitalisation
    text = _capitalise_sentences(text)

    return text.strip()
