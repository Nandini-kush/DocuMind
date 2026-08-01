import os
import logging
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_tokenizer = None
_model = None
_model_lock = threading.Lock()

FALLBACK_RESPONSE = "The document does not contain this information."
FALLBACK_STRING = FALLBACK_RESPONSE


def load_model_and_tokenizer():
    """Thread-safe lazy initializer for FLAN-T5 model and tokenizer."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    with _model_lock:
        if _tokenizer is None or _model is None:
            model_names = ["google/flan-t5-base", "google/flan-t5-small"]
            for model_name in model_names:
                try:
                    logger.info(f"[LLM-INIT] Attempting to load '{model_name}' (local_files_only=True)...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
                    _tokenizer = tokenizer
                    _model = model
                    logger.info(f"[LLM-INIT] Successfully loaded '{model_name}'.")
                    return _tokenizer, _model
                except Exception as e:
                    logger.warning(f"[LLM-INIT] Failed local load for '{model_name}': {e}. Retrying without local restriction...")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                        _tokenizer = tokenizer
                        _model = model
                        logger.info(f"[LLM-INIT] Successfully loaded '{model_name}' via standard load.")
                        return _tokenizer, _model
                    except Exception as e2:
                        logger.error(f"[LLM-INIT] Standard load failed for '{model_name}': {e2}")

            logger.critical("[LLM-INIT] All FLAN-T5 model initialization attempts failed.")
            _tokenizer = None
            _model = None

    return _tokenizer, _model


class PipelineAdapter:
    """Backward-compatible adapter matching the call signature expected by legacy tests."""
    def __call__(self, prompt: str, do_sample: bool = False, max_new_tokens: int = 150):
        tokenizer, model = load_model_and_tokenizer()
        if tokenizer is None or model is None:
            raise RuntimeError("FLAN-T5 model is not available.")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": generated_text}]


def get_qa_pipeline():
    """Thread-safe getter for QA pipeline adapter."""
    tokenizer, model = load_model_and_tokenizer()
    if tokenizer is None or model is None:
        return None
    return PipelineAdapter()


def format_extractive_prompt(query: str, context_chunks) -> str:
    """Formats context chunks into an extractive prompt template for FLAN-T5."""
    if not context_chunks:
        return ""

    if isinstance(context_chunks, str):
        context_chunks = [context_chunks]

    context_blocks = []
    for i, chunk in enumerate(context_chunks):
        if isinstance(chunk, dict):
            text_val = chunk.get("text", "")
        else:
            text_val = str(chunk)
        if text_val.strip():
            context_blocks.append(f"[Source {i+1}]: {text_val.strip()}")

    if not context_blocks:
        return ""

    context_str = "\n\n".join(context_blocks)
    if len(context_str) > 2000:
        context_str = context_str[:2000] + "..."

    prompt = (
        "Extract the answer ONLY from the Context below. "
        "Do not use external knowledge or invent facts. "
        "If the document context does not explicitly contain the answer, reply exactly: 'The document does not contain this information.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt


format_prompt = format_extractive_prompt


def clean_answer_output(answer: str) -> str:
    """Post-processes LLM output to eliminate hallucination indicators."""
    if not answer or not answer.strip():
        return FALLBACK_RESPONSE

    cleaned = answer.strip()
    missing_phrases = [
        "not contain", "not mentioned", "not explicitly", "no information",
        "does not state", "cannot answer", "cannot find",
        "unspecified", "don't know", "i don't", "no answer"
    ]

    for phrase in missing_phrases:
        if phrase in cleaned.lower():
            logger.info(f"[LLM-SCRUB] Missing phrase detected ('{phrase}'). Standardizing to fallback string.")
            return FALLBACK_RESPONSE

    return cleaned


class AnswerResult(str):
    """String subclass that allows tuple unpacking (answer, sources) for API handlers."""
    def __new__(cls, answer_text, sources=None):
        obj = super().__new__(cls, answer_text)
        obj.sources = sources or []
        return obj

    def __iter__(self):
        yield str(self)
        yield self.sources


def generate_answer(query: str, context_chunks) -> AnswerResult:
    """Generates extractive answer using FLAN-T5 model pipeline."""
    if isinstance(context_chunks, str):
        context_chunks = [context_chunks]

    valid_chunks = []
    if context_chunks:
        for c in context_chunks:
            text_val = c.get("text", "") if isinstance(c, dict) else str(c)
            if text_val.strip():
                valid_chunks.append(text_val.strip())

    if not valid_chunks:
        logger.info("[LLM-GEN] Empty context provided. Returning fallback response.")
        return AnswerResult(FALLBACK_RESPONSE, [])

    prompt = format_extractive_prompt(query, valid_chunks)

    # LOG STAGE 9: Prompt Sent to LLM
    logger.info("=== [RETRIEVAL STAGE 9] Prompt Sent to LLM ===")
    logger.info(f"\n{prompt}\n")

    pipe = get_qa_pipeline()
    if pipe is None:
        logger.error("[LLM-GEN] QA model is unavailable.")
        return AnswerResult(FALLBACK_RESPONSE, [])

    try:
        response = pipe(prompt, do_sample=False, max_new_tokens=150)
        raw_answer = response[0]["generated_text"]
        logger.info(f"[LLM-GEN] Raw model output: '{raw_answer}'")

        final_answer = clean_answer_output(raw_answer)
        sources = [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(valid_chunks[:3])]

        # LOG STAGE 10: Final LLM Answer
        logger.info("=== [RETRIEVAL STAGE 10] Final LLM Answer ===")
        logger.info(f"  ANSWER: '{final_answer}'\n")

        return AnswerResult(final_answer, sources)

    except Exception as e:
        logger.error(f"[LLM-GEN] Error during generation: {e}")
        return AnswerResult(FALLBACK_RESPONSE, [])


answer_question = generate_answer
