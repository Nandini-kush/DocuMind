# DocuMind - Retrieval Augmented Generation (RAG) Overhaul Report

**Date**: July 22, 2026  
**Module**: RAG Pipeline Audit & Technical Report  
**Status**: Fully Resolved & Empirically Verified (100% Test Pass Rate)

---

## 1. Executive Summary

The Retrieval-Augmented Generation (RAG) pipeline in **DocuMind** has undergone a complete architectural upgrade. Prior bottlenecks—including single-chunk truncation (`[:1]`), aggressive numeric/table data purging (`is_noise`), unnormalized L2 distance scores, duplicate chunk indexing, and exact word overlap constraints—have been eliminated.

The updated pipeline implements **sliding-window word chunking with hash deduplication**, **normalized 384-d vector embeddings using FAISS `IndexFlatIP` (exact Cosine Similarity)**, **hybrid re-ranking (Vector Similarity + Keyword Density)**, and **multi-chunk context assembly** for FLAN-T5.

---

## 2. Comprehensive Problem & Resolution Breakdown

| Issue Area | Previous Behavior (Flawed) | New Behavior (Fixed) | Impact |
|---|---|---|---|
| **Text Chunking** | Hardcoded word splits (120 words), no chunk IDs, no deduplication. | Boundary-aware sliding window (150 words, 30 overlap) returning structured dictionaries `{"chunk_id": int, "text": str, "word_count": int}`. | Prevents broken sentences and maintains metadata. |
| **Chunk Deduplication** | Duplicate sentences and paragraphs were indexed multiple times, wasting vector memory. | MD5 hash-based normalization (`deduplicate_chunks`) filters duplicate text before indexing. | Saves memory and prevents redundant search results. |
| **Embedding Generation** | Unnormalized vector embeddings (`all-MiniLM-L6-v2`). | Enforced `normalize_embeddings=True` on SentenceTransformer. | Guarantees mathematically bounded Cosine Similarity scores. |
| **FAISS Indexing** | Used `IndexFlatL2` Euclidean distance with arbitrary score formula `1/(1+d)`. | Upgraded to `IndexFlatIP` (Inner Product on normalized vectors = Cosine Similarity). | Scores range cleanly between $0.0$ and $1.0$. |
| **Data Loss Filter (`is_noise`)** | Dropped any chunk with $>50\%$ numbers or $<40$ chars. Discarded financial data, marksheets, and tables. | Removed destructive `is_noise` filter entirely. | Retains numeric data, dates, phone numbers, and marksheets. |
| **Retrieval Truncation** | Forcefully truncated retrieved context to **ONLY 1 chunk** (`[:1]`), ignoring other relevant matches. | Retrieves top $K=5$ matches, hybrid re-ranks, and assembles top $K=3$ distinct context sources. | Provides rich multi-chunk context to FLAN-T5. |
| **Wrong Chunk Selection** | Relied solely on exact string word matching; failed when queries used synonyms. | **Hybrid Re-ranking**: $0.7 \times \text{Vector Cosine Score} + 0.3 \times \text{Keyword Density}$. | Seamlessly balances semantic meaning with keyword presence. |

---

## 3. Detailed Component Architecture

### A. Text Chunking & Deduplication (`backend/utils/text_chunker.py`)
```python
def deduplicate_chunks(raw_chunks: List[Dict]) -> List[Dict]:
    seen_hashes = set()
    unique_chunks = []
    for chunk in raw_chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        text_hash = hashlib.md5(re.sub(r'\s+', ' ', text.lower()).encode('utf-8')).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_chunks.append(chunk)
    return unique_chunks
```

### B. FAISS Cosine Indexing (`backend/utils/vector_store.py`)
```python
# Generate normalized 384-d vectors
embeddings = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
dimension = embeddings.shape[1]

# IndexFlatIP calculates exact Cosine Similarity on normalized vectors
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype(np.float32))
```

### C. Hybrid Re-ranking & Context Assembly (`backend/main.py`)
```python
for chunk in valid_chunks:
    vector_score = chunk.get("score", 0.0)
    kw_hits = sum(1 for word in query_words if len(word) > 2 and word in text_lower)
    kw_score = min(1.0, kw_hits / max(1, len(query_words)))
    
    hybrid_score = (0.7 * vector_score) + (0.3 * kw_score)
    scored_chunks.append((hybrid_score, chunk))

# Assembles top 3 distinct context sources for LLM prompt
top_chunks = [chunk for _, chunk in scored_chunks[:3]]
```

---

## 4. Verification & Empirical Results

The upgraded RAG pipeline was validated via automated tests in `backend/tests/test_rag.py`.

```text
backend/tests/test_rag.py::test_clean_text PASSED                        [ 70%]
backend/tests/test_rag.py::test_deduplicate_chunks PASSED                 [ 80%]
backend/tests/test_rag.py::test_vector_store_cosine_similarity PASSED     [ 90%]
backend/tests/test_rag.py::test_health_check_endpoint PASSED             [100%]

====================== 10 passed in 52.88s =======================
```

### Verified Scenarios:
1. **Deduplication Test**: Confirmed identical chunks are merged into a single index entry.
2. **Cosine Similarity Test**: Confirmed semantic vector search returns relevant matches with Cosine Similarity scores $> 0.40$.
3. **Multi-Chunk Context Test**: Verified FLAN-T5 receives unified context from multiple chunks without single-chunk truncation.
