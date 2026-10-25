import re
import unicodedata


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).strip()).lower()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))
