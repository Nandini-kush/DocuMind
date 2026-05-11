from transformers import pipeline
import torch

# ✅ Use text2text-generation for FLAN-T5
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=100,      # ⬅ reduce verbosity
    do_sample=False,         # ⬅ deterministic answers
    temperature=0.0,         # ⬅ prevents creativity/hallucination
    repetition_penalty=1.2   # ⬅ avoids repeating phrases
)


def generate_answer(question, context_chunks):
    context = "\n".join(context_chunks)

    prompt = f"""You are a document question answering assistant.

Answer the question ONLY using the information provided in the context.

Rules:
- Extract only the information necessary to answer the question.
- Do NOT include unrelated information.
- Do NOT include document titles, headers, footers, metadata, or section labels unless they are directly relevant.
- Keep the answer concise and factual.
- Prefer one or two sentences, or a short bullet list if the question asks for multiple items.
- If the answer is not present in the context, say exactly:
  "The document does not contain this information."

Context:
{context}

Question:
{question}

Answer:"""


    result = qa_pipeline(prompt)

    answer_text = result[0]["generated_text"]

    # remove prompt echo if model repeats it
    answer_text = answer_text.replace(prompt, "").strip()

    import re
    # Output Post-Processing: Remove repeated prefixes
    answer_text = re.sub(r'^(Answer|Output|Result|Note):\s*', '', answer_text, flags=re.IGNORECASE)
    
    # Remove obvious leftover metadata markers from answer
    answer_text = re.sub(r'(?i)^(page \d+|confidential|internal use|title:|header:|footer:)\s*', '', answer_text).strip()

    if not answer_text or len(answer_text) < 5 or "document does not contain" in answer_text.lower():
        return "The document does not contain this information."

    return answer_text

