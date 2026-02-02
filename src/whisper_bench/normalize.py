"""Text normalization for fair WER/CER comparison."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison.

    - Unicode NFC normalization (important for Indonesian)
    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation (keep letters, numbers, whitespace)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text
