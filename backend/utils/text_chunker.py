import re
import hashlib
from typing import List, Dict, Any, Union


def clean_text(text: str) -> str:
    """Cleans extracted text while preserving section structure, newlines, and key-values."""
    if not text:
        return ""

    # Remove non-printable control characters except newline
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Remove repetitive page numbers e.g. "Page 1 of 5" or "Page 2"
    cleaned = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', cleaned)

    # Replace multiple horizontal spaces/tabs with single space (preserve newlines)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)

    # Replace 4 or more consecutive newlines with 3 newlines for clean paragraph separation
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)

    return cleaned.strip()


def extract_document_header(text: str) -> str:
    """Extracts top document header context (first lines / candidate info)."""
    if not text or not text.strip():
        return ""

    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return ""

    header_lines = lines[:3]
    header_summary = " | ".join(header_lines)
    if len(header_summary) > 250:
        header_summary = header_summary[:250] + "..."

    return header_summary


class RecursiveCharacterChunker:
    """
    Section-aware recursive character chunker.
    Splits text hierarchically using separator priorities: ["\n\n\n", "\n\n", "\n", ". ", " ", ""].
    Preserves document structure and attaches metadata to each chunk.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n\n", "\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text into chunks bounded by target chunk_size."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits = []
        for s in splits:
            if not s.strip():
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merges small splits with overlap while respecting chunk_size bounds."""
        docs = []
        current_doc = []
        total = 0

        for d in splits:
            len_d = len(d)
            if total + len_d + (len(separator) if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc_text = separator.join(current_doc).strip()
                    if doc_text:
                        docs.append(doc_text)
                    while total > self.chunk_overlap and current_doc:
                        removed = current_doc.pop(0)
                        total -= (len(removed) + len(separator))
                current_doc = [d]
                total = len_d
            else:
                current_doc.append(d)
                total += len_d + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc_text = separator.join(current_doc).strip()
            if doc_text:
                docs.append(doc_text)

        return docs

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Chunks document text into structured metadata objects."""
        cleaned = clean_text(text)
        if not cleaned:
            return []

        header_summary = extract_document_header(cleaned)
        raw_chunks = self._split_text(cleaned, self.separators)

        chunk_objects = []
        for idx, chunk_str in enumerate(raw_chunks):
            chunk_text = chunk_str.strip()
            if not chunk_text:
                continue

            is_header = (idx == 0)
            section_name = "Header" if is_header else "Body"

            lower_chunk = chunk_text.lower()
            if "education" in lower_chunk:
                section_name = "Education"
            elif "experience" in lower_chunk or "employment" in lower_chunk or "work history" in lower_chunk:
                section_name = "Experience"
            elif "project" in lower_chunk or "projects" in lower_chunk:
                section_name = "Projects"
            elif "skill" in lower_chunk or "technologies" in lower_chunk:
                section_name = "Skills"

            chunk_objects.append({
                "chunk_id": idx,
                "text": chunk_text,
                "section": section_name,
                "is_header": is_header,
                "header_summary": header_summary
            })

        return chunk_objects


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """Backward-compatible entry point for section-aware recursive chunking."""
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    return chunker.chunk(text)


def deduplicate_chunks(chunks: List[Union[Dict[str, Any], str]]) -> List[Union[Dict[str, Any], str]]:
    """Deduplicates text chunks using MD5 hash comparison while preserving metadata structure and order."""
    if not chunks:
        return []

    seen_hashes = set()
    unique = []

    for item in chunks:
        if isinstance(item, dict):
            text_val = item.get("text", "")
        else:
            text_val = str(item)

        norm_text = text_val.strip().lower()
        if not norm_text:
            continue

        chunk_hash = hashlib.md5(norm_text.encode("utf-8")).hexdigest()
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            unique.append(item)

    return unique
