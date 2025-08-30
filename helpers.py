import re, unicodedata

def safe_filename(name: str) -> str:
    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:255]
