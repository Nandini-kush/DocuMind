import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm import (
    format_prompt,
    clean_answer_output,
    answer_question,
    generate_answer,
    FALLBACK_STRING
)

def test_format_prompt():
    prompt = format_prompt("What is DocuMind?", ["DocuMind is an AI document chatbot.", "It features OCR and FAISS."])
    assert "[Source 1]: DocuMind is an AI document chatbot." in prompt
    assert "[Source 2]: It features OCR and FAISS." in prompt
    assert "Question: What is DocuMind?" in prompt
    assert "Extract the answer ONLY from the Context" in prompt

def test_clean_answer_output():
    # Valid answer
    assert clean_answer_output("DocuMind is a chatbot") == "DocuMind is a chatbot"
    
    # Anti-hallucination missing indicators
    assert clean_answer_output("The document does not contain this") == FALLBACK_STRING
    assert clean_answer_output("Not mentioned in context") == FALLBACK_STRING
    assert clean_answer_output("I don't know") == FALLBACK_STRING
    assert clean_answer_output("Cannot answer from text") == FALLBACK_STRING
    assert clean_answer_output("") == FALLBACK_STRING

def test_generate_answer_empty_context():
    assert generate_answer("What is the revenue?", []) == FALLBACK_STRING
    assert generate_answer("What is the revenue?", ["   "]) == FALLBACK_STRING

@patch("utils.llm.get_qa_pipeline")
def test_answer_question_mocked_pipeline(mock_get_pipeline):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"generated_text": "DocuMind processes documents using RAG and OCR."}]
    mock_get_pipeline.return_value = mock_pipeline

    context = "DocuMind processes documents using RAG and OCR."
    answer = answer_question("How does DocuMind process documents?", context)
    assert answer == "DocuMind processes documents using RAG and OCR."

