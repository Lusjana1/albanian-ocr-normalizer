"""
Rule-based Albanian text cleaner.
Handles OCR noise, character substitutions, spacing, and
Albanian-specific orthographic patterns.
"""
import re


# ---------------------------------------------------------------------------
# OCR character substitution corrections (context-free)
# ---------------------------------------------------------------------------
_CHAR_SUBS = [
    # Digits mistaken for letters
    (r"(?<=[a-zA-ZëçËÇ])0(?=[a-zA-ZëçËÇ])", "o"),
    (r"(?<=[a-zA-ZëçËÇ])1(?=[a-zA-ZëçËÇ])", "i"),
    (r"\b1(?=[a-zA-ZëçËÇ])", "I"),
    # Common glyph confusions
    (r"rn", "m"),          # rn → m (very common OCR error)
    (r"li(?=[a-z])", "h"), # 'li' → 'h' only at word interior (heuristic)
    # Pipe/broken bar used as 'i' or 'l'
    (r"\|(?=[a-z])", "l"),
]

# ---------------------------------------------------------------------------
# Albanian digraph / special-character corrections
# ---------------------------------------------------------------------------
_ALBANIAN_FIXES = [
    # é, è, ê → ë (common OCR mistake for Albanian ë)
    (r"[éèê]", "ë"),
    # Ç / Ç variations
    (r"[Ćć]", "ç"),
    # Fix 'dh', 'gj', 'nj', 'sh', 'th', 'xh', 'zh' split by a space/hyphen
    (r"(?i)\b(d)\s*-?\s*(h)\b", r"\1\2"),
    (r"(?i)\b(g)\s*-?\s*(j)\b", r"\1\2"),
    (r"(?i)\b(n)\s*-?\s*(j)\b", r"\1\2"),
    (r"(?i)\b(s)\s*-?\s*(h)\b", r"\1\2"),
    (r"(?i)\b(t)\s*-?\s*(h)\b", r"\1\2"),
    (r"(?i)\b(x)\s*-?\s*(h)\b", r"\1\2"),
    (r"(?i)\b(z)\s*-?\s*(h)\b", r"\1\2"),
]

# ---------------------------------------------------------------------------
# Noise removal
# ---------------------------------------------------------------------------
_NOISE_PATTERNS = [
    # Sequences of 3+ repeated non-alphanumeric chars
    (r"([^a-zA-Z0-9ëçËÇ\s])\1{2,}", ""),
    # Lone single non-alphanumeric/non-punctuation symbols on their own line
    (r"(?m)^[^a-zA-Z0-9ëçËÇ]+$", ""),
    # Page numbers: standalone digit sequences on their own line
    (r"(?m)^\s*\d{1,4}\s*$", ""),
    # Excessive dashes used as separators
    (r"-{3,}", ""),
    # Remove zero-width / control characters except newlines/tabs
    (r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", ""),
]

# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------
_SPACE_PATTERNS = [
    (r"[ \t]+", " "),           # Multiple spaces/tabs → single space
    (r" +\n", "\n"),            # Trailing spaces before newline
    (r"\n{3,}", "\n\n"),        # Max two consecutive newlines
    (r"^\s+|\s+$", ""),         # Strip leading/trailing whitespace
]

# ---------------------------------------------------------------------------
# Punctuation fixes
# ---------------------------------------------------------------------------
_PUNCT_PATTERNS = [
    # Space before punctuation → no space
    (r"\s+([,.!?;:»])", r"\1"),
    # No space after opening quote/bracket → add space
    (r"(«|\(|\[)\s*", r"\1"),
    # Ensure space after sentence-ending punctuation
    (r"([.!?])([A-ZËÇА-Я])", r"\1 \2"),
    # Fix double punctuation
    (r"([.!?]){2,}", r"\1"),
    # Straight quotes → Albanian guillemets (optional, helps readability)
    (r'"([^"]*)"', r"«\1»"),
]


def _apply_patterns(text: str, patterns: list) -> str:
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    return text


def _capitalize_sentences(text: str) -> str:
    """Capitalize the first letter of each sentence."""
    def cap(m):
        return m.group(0)[0] + m.group(0)[1].upper() + m.group(0)[2:]

    text = re.sub(r"(^|[.!?]\s+)([a-zëç])", lambda m: m.group(1) + m.group(2).upper(), text)
    # Capitalize very first character
    if text:
        text = text[0].upper() + text[1:]
    return text


def clean_text(raw: str) -> str:
    """Full cleaning pipeline."""
    text = raw

    # 1. OCR character substitutions
    text = _apply_patterns(text, _CHAR_SUBS)

    # 2. Albanian-specific fixes
    text = _apply_patterns(text, _ALBANIAN_FIXES)

    # 3. Remove noise
    text = _apply_patterns(text, _NOISE_PATTERNS)

    # 4. Normalise whitespace
    text = _apply_patterns(text, _SPACE_PATTERNS)

    # 5. Fix punctuation
    text = _apply_patterns(text, _PUNCT_PATTERNS)

    # 6. Capitalise sentences
    text = _capitalize_sentences(text)

    return text.strip()
