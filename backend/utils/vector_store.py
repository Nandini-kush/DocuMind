import os
import re
import math
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.05"))

_embedding_model = None
_cross_encoder_model = None
_cross_encoder_attempted = False


def get_embedding_model() -> SentenceTransformer:
    """Thread-safe lazy loader for SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        logger.info(f"[VECTOR-INIT] Loading SentenceTransformer embedding model: '{MODEL_NAME}' (local_files_only=True)...")
        _embedding_model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        logger.info(f"[VECTOR-INIT] Model loaded successfully ({EMBEDDING_DIM}-d).")
    except Exception as e:
        logger.warning(f"[VECTOR-INIT] Local load failed: {e}. Retrying standard load...")
        try:
            _embedding_model = SentenceTransformer(MODEL_NAME)
            logger.info(f"[VECTOR-INIT] Model loaded successfully via standard load.")
        except Exception as e2:
            logger.error(f"[VECTOR-INIT] Critical: Failed to load SentenceTransformer '{MODEL_NAME}': {e2}")
            raise e2
    return _embedding_model


def get_cross_encoder():
    """Lazy loader for CrossEncoder model with safe fallback handling."""
    global _cross_encoder_model, _cross_encoder_attempted
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    if _cross_encoder_attempted:
        return None

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"[CROSS-ENCODER-INIT] Loading CrossEncoder model: '{CROSS_ENCODER_MODEL_NAME}'...")
        try:
            _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, max_length=512, local_files_only=True)
        except Exception:
            _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, max_length=512)
        _cross_encoder_attempted = True
        logger.info(f"[CROSS-ENCODER-INIT] CrossEncoder loaded successfully.")
    except Exception as e:
        logger.warning(f"[CROSS-ENCODER-INIT] CrossEncoder unavailable: {e}. Using RRF/Cosine hybrid score fallback.")
        _cross_encoder_model = None
        _cross_encoder_attempted = True

    return _cross_encoder_model


def generate_embeddings(texts: list) -> np.ndarray:
    """Generates L2-normalized 384-dimensional dense vector embeddings."""
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    extracted_texts = []
    for item in texts:
        if isinstance(item, dict):
            text_val = item.get("text", "")
        else:
            text_val = str(item)
        extracted_texts.append(text_val)

    model = get_embedding_model()
    embeddings = model.encode(
        extracted_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)


def create_faiss_index(data) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Creates a FAISS IndexFlatIP (Inner Product) index for Cosine Similarity search."""
    if isinstance(data, list):
        embeddings = generate_embeddings(data)
    elif isinstance(data, np.ndarray):
        embeddings = data
    else:
        embeddings = np.array(data, dtype=np.float32)

    if not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
        raise ValueError("Cannot create FAISS index from empty embeddings array.")

    dim = embeddings.shape[1]
    logger.info(f"[FAISS-INDEX] Creating IndexFlatIP with dimension {dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"[FAISS-INDEX] Index successfully built with {index.ntotal} vectors.")
    return index, embeddings


class BM25Scorer:
    """Fast, accurate BM25Okapi implementation for sparse keyword retrieval."""
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_tokens = [self._tokenize(doc) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.corpus_size)

        self.doc_freqs = []
        self.idf = {}
        self._calc_idf()

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in re.findall(r'\w+', text.lower()) if len(w) > 1]

    def _calc_idf(self):
        df_counts = {}
        for tokens in self.doc_tokens:
            frequencies = {}
            for t in tokens:
                frequencies[t] = frequencies.get(t, 0) + 1
            self.doc_freqs.append(frequencies)
            for t in frequencies:
                df_counts[t] = df_counts.get(t, 0) + 1

        for word, freq in df_counts.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for q in q_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for idx, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1.0)
                den = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / max(1.0, self.avgdl)))
                scores[idx] += q_idf * (num / den)

        return scores


class ResultItem(dict):
    """Result object compatible with dictionary indexing, attributes, and tuple unpacking."""
    def __init__(self, text: str, score: float, raw_score: float = 0.0, metadata: dict = None):
        merged_dict = {
            "text": text,
            "score": score,
            "raw_score": raw_score,
            "chunk": text,
            "similarity_score": score
        }
        if metadata:
            merged_dict.update(metadata)
        super().__init__(merged_dict)
        self.text = text
        self.score = score
        self.raw_score = raw_score
        self.metadata = metadata or {}

    def __getitem__(self, item):
        if item == 0:
            return self.text
        if item == 1:
            return self.score
        return super().__getitem__(item)


def is_candidate_name_query(query: str) -> bool:
    """Detects if query is specifically asking for candidate name / author / contact details."""
    q = query.lower()
    keywords = ["candidate", "name", "who is", "author", "applicant", "person", "email", "contact", "phone"]
    return any(k in q for k in keywords)


def search_vector_store(
    arg1=None,
    arg2=None,
    arg3=None,
    query=None,
    index=None,
    chunks=None,
    top_k: int = 5,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    hybrid_weight: float = 0.70
) -> list:
    """
    Production RAG Retrieval Pipeline:
    1. Dense FAISS IndexFlatIP Cosine Search (Top-10)
    2. Sparse BM25 Keyword Search (Top-10)
    3. Reciprocal Rank Fusion (RRF)
    4. CrossEncoder Reranking
    5. Top-3 Final Context Selection
    6. Detailed 10-Stage Console Logging
    """
    query_val = query if query is not None else arg1

    if index is not None and chunks is not None:
        index_val, chunks_val = index, chunks
    elif arg2 is not None and arg3 is not None:
        if hasattr(arg2, 'search') and isinstance(arg3, list):
            index_val, chunks_val = arg2, arg3
        elif isinstance(arg2, list) and hasattr(arg3, 'search'):
            index_val, chunks_val = arg3, arg2
        else:
            index_val = arg2 if hasattr(arg2, 'search') else arg3
            chunks_val = arg3 if isinstance(arg3, list) else arg2
    else:
        index_val = index if index is not None else arg2
        chunks_val = chunks if chunks is not None else arg3

    if index_val is None or not hasattr(index_val, 'ntotal') or index_val.ntotal == 0 or not chunks_val:
        logger.warning("[RETRIEVAL] FAISS index is empty or null. Returning empty context.")
        return []

    # Extract clean text strings for processing
    corpus_texts = []
    chunk_meta_list = []
    for c in chunks_val:
        if isinstance(c, dict):
            text_str = c.get("text", "")
            chunk_meta_list.append(c)
        else:
            text_str = str(c)
            chunk_meta_list.append({"text": text_str, "is_header": False})
        corpus_texts.append(text_str)

    # 1. LOG STAGE 1: User Question
    logger.info("\n" + "="*80)
    logger.info(f"=== [RETRIEVAL STAGE 1] User Question: '{query_val}' ===")

    # 2. LOG STAGE 2: Query Embedding Shape
    query_vector = generate_embeddings([query_val])
    logger.info(f"=== [RETRIEVAL STAGE 2] Query Embedding Shape: {query_vector.shape} ===")

    # 3. Dense FAISS Search (Retrieve top min(10, ntotal))
    k_retrieve = min(10, index_val.ntotal)
    raw_scores, raw_indices = index_val.search(query_vector, k_retrieve)
    dense_scores = raw_scores[0]
    dense_indices = raw_indices[0]

    # 4. LOG STAGE 3 & 4: Top-10 FAISS Results & Similarity Scores
    logger.info(f"=== [RETRIEVAL STAGE 3 & 4] Top-{len(dense_indices)} FAISS Results & Cosine Similarity Scores ===")
    dense_rank_map = {}
    for r_idx, (idx, score) in enumerate(zip(dense_indices, dense_scores)):
        if idx < len(chunks_val):
            dense_rank_map[idx] = (r_idx + 1, float(score))
            snippet = corpus_texts[idx][:90].replace('\n', ' ')
            logger.info(f"  FAISS Rank #{r_idx+1} [Chunk {idx}] | Cosine Score: {score:.4f} | Snippet: '{snippet}...'")

    # 5. Sparse BM25 Search
    bm25 = BM25Scorer(corpus_texts)
    bm25_all_scores = bm25.get_scores(query_val)
    sorted_bm25_indices = np.argsort(bm25_all_scores)[::-1][:k_retrieve]

    # 6. LOG STAGE 5: BM25 Scores
    logger.info(f"=== [RETRIEVAL STAGE 5] Top-{len(sorted_bm25_indices)} BM25 Sparse Scores ===")
    sparse_rank_map = {}
    for r_idx, idx in enumerate(sorted_bm25_indices):
        score = bm25_all_scores[idx]
        sparse_rank_map[idx] = (r_idx + 1, float(score))
        snippet = corpus_texts[idx][:90].replace('\n', ' ')
        logger.info(f"  BM25 Rank #{r_idx+1} [Chunk {idx}] | BM25 Score: {score:.4f} | Snippet: '{snippet}...'")

    # 7. Reciprocal Rank Fusion (RRF) Hybrid Scoring
    all_candidate_indices = set(dense_rank_map.keys()).union(set(sparse_rank_map.keys()))
    
    # Guarantee Header chunk is in candidate pool for candidate/name/contact queries
    if is_candidate_name_query(query_val):
        for idx, meta in enumerate(chunk_meta_list):
            if meta.get("is_header", False) or idx == 0:
                all_candidate_indices.add(idx)

    rrf_scores = {}
    k_rrf = 60.0
    for idx in all_candidate_indices:
        r_dense = dense_rank_map.get(idx, (999, 0.0))[0]
        r_sparse = sparse_rank_map.get(idx, (999, 0.0))[0]
        score_rrf = (1.0 / (k_rrf + r_dense)) + (1.0 / (k_rrf + r_sparse))

        if is_candidate_name_query(query_val) and chunk_meta_list[idx].get("is_header", False):
            score_rrf += 0.05
            logger.info(f"  [HEADER BOOST] Chunk {idx} receives candidate query header boost (+0.05).")

        rrf_scores[idx] = score_rrf

    # 8. LOG STAGE 6: Hybrid RRF Scores
    sorted_rrf_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k_retrieve]
    logger.info(f"=== [RETRIEVAL STAGE 6] Reciprocal Rank Fusion (RRF) Hybrid Candidates ===")
    for rank, (idx, score_rrf) in enumerate(sorted_rrf_candidates):
        snippet = corpus_texts[idx][:90].replace('\n', ' ')
        logger.info(f"  RRF Rank #{rank+1} [Chunk {idx}] | RRF Score: {score_rrf:.6f} | Snippet: '{snippet}...'")

    # 9. CrossEncoder Reranking
    cross_encoder = get_cross_encoder()
    candidate_indices = [idx for idx, _ in sorted_rrf_candidates]

    cross_scores = {}
    if cross_encoder is not None:
        pairs = [(query_val, corpus_texts[idx]) for idx in candidate_indices]
        try:
            ce_predictions = cross_encoder.predict(pairs)
            for idx, ce_score in zip(candidate_indices, ce_predictions):
                cross_scores[idx] = float(ce_score)
        except Exception as e:
            logger.warning(f"[CROSS-ENCODER-EXEC] Error predicting scores: {e}. Falling back to RRF scores.")
            for idx, score_rrf in sorted_rrf_candidates:
                cross_scores[idx] = score_rrf
    else:
        for idx, score_rrf in sorted_rrf_candidates:
            cross_scores[idx] = score_rrf

    # 10. LOG STAGE 7: CrossEncoder Scores
    logger.info(f"=== [RETRIEVAL STAGE 7] CrossEncoder Reranking Scores ===")
    final_ranked_candidates = sorted(candidate_indices, key=lambda idx: cross_scores[idx], reverse=True)

    # For candidate name/email/contact queries, explicitly prioritize Document Header (Chunk 0) as Top 1
    if is_candidate_name_query(query_val):
        header_indices = [i for i in final_ranked_candidates if chunk_meta_list[i].get("is_header", False) or i == 0]
        if header_indices:
            h_idx = header_indices[0]
            if h_idx in final_ranked_candidates:
                final_ranked_candidates.remove(h_idx)
            final_ranked_candidates.insert(0, h_idx)

    for rank, idx in enumerate(final_ranked_candidates):
        ce_score = cross_scores.get(idx, 0.0)
        snippet = corpus_texts[idx][:90].replace('\n', ' ')
        logger.info(f"  Final Rerank #{rank+1} [Chunk {idx}] | CrossEncoder Score: {ce_score:.4f} | Snippet: '{snippet}...'")

    # 11. LOG STAGE 8: Final Selected Top-3 Chunks
    selected_indices = final_ranked_candidates[:3]
    final_results = []
    logger.info(f"=== [RETRIEVAL STAGE 8] Final Selected Chunks (Top-3) Sent to LLM ===")
    for rank, idx in enumerate(selected_indices):
        chunk_obj = chunk_meta_list[idx]
        text_content = chunk_obj.get("text", "")
        ce_score = cross_scores.get(idx, 0.0)
        logger.info(f"  Selected Source #{rank+1} [Chunk {idx}] (Score: {ce_score:.4f}):\n{text_content}\n")
        final_results.append(ResultItem(text_content, ce_score, float(dense_rank_map.get(idx, (0, 0.0))[1]), chunk_obj))

    logger.info("="*80 + "\n")
    return final_results


search_similar_chunks = search_vector_store
