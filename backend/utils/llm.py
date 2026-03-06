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
You are a document question answering system.

Answer the question ONLY using the information in the context.

Rules:
- Extract only the relevant sentence.
- Do NOT include unrelated topics.
- Do NOT include section numbers unless asked.
- Do NOT include institute names or document metadata.
- Keep the answer short (maximum 2 sentences).

If the answer is not present, say exactly:
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

