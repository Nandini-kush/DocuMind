import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def chunk_text(text, chunk_size=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append({"text": current_chunk.strip()})
            current_chunk = sentence

    if current_chunk:
        chunks.append({"text": current_chunk.strip()})

    return chunks
