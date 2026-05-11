import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove common page numbers and standalone metadata
    text = re.sub(r'(?i)\bpage\s+\d+\b(\s+of\s+\d+)?', '', text)
    text = re.sub(r'^\s*[-_]*\s*\d+\s*[-_]*\s*$', '', text, flags=re.MULTILINE)

    # Remove excessive newlines and whitespace
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

    if not words:
        return chunks

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunks.append({"text": chunk_str})

        if i + chunk_size >= len(words):
            break

        i += (chunk_size - overlap)

    return chunks
