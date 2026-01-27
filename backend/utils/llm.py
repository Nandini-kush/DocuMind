from transformers import pipeline
import torch

# ✅ Use text2text-generation for FLAN-T5
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=150,      # ⬅ reduce verbosity
    do_sample=False,         # ⬅ deterministic answers
    temperature=0.0,         # ⬅ prevents creativity/hallucination
    repetition_penalty=1.2   # ⬅ avoids repeating phrases
)


def generate_answer(question, context_chunks):
    context = "\n".join(context_chunks)

    prompt = f"""
You are a strict question-answering assistant.

Answer the question using ONLY the information provided in the context below.

Rules:
- Do NOT include section numbers unless explicitly asked.
- Do NOT include unrelated topics.
- Do NOT include institute names, exam names, or document metadata.
- Be concise and precise.
- If the answer is partially present, answer ONLY what is present.
- If the answer is NOT present, say exactly:
"The document does not contain this information."

Context:
{context}

Question:
{question}

Answer:
"""

    result = qa_pipeline(prompt)

    answer_text = result[0]["generated_text"]

    # remove prompt echo if model repeats it
    answer_text = answer_text.replace(prompt, "").strip()

    return answer_text

