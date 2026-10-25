"""Visible CER policy; no OpenCC, numeric rewriting or filler removal."""
import re
import unicodedata

NORMALIZATION_VERSION = "nfkc-lower-unicode-punctuation-whitespace-v1"


def normalize_for_cer(text: str) -> str:
    """NFKC, strip/collapse whitespace, lowercase, remove P* and whitespace.

    Asterisk is Unicode punctuation: it is removed from CER but separately
    counted in every record. Symbols such as +, %, currency follow Unicode
    categories (percent is P*, plus/currency are S* and remain).
    Chinese, traditional characters, digits and spoken fillers are retained.
    """
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip()).lower()
    return "".join(c for c in text if not c.isspace()
                   and not unicodedata.category(c).startswith("P"))


def asterisk_count(text: str) -> int:
    return unicodedata.normalize("NFKC", text).count("*")
