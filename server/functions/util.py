
# --- Utility Functions ---
import io
from typing import List, Dict
from PyPDF2 import PdfReader
from fastapi import UploadFile
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

def get_text_from_pdf(file: UploadFile) -> str:
    """
    Extract text from PDF safely. Handles pages where extract_text() may return None.
    """
    pdf_reader = PdfReader(io.BytesIO(file.file.read()))
    text_parts = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into fixed-size chunks (no overlap)."""
    if not text:
        return []
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))
        chunks.append(text[current_pos:end_pos])
        current_pos += chunk_size
    return chunks

def get_llm_response(prompt: str) -> str:
    """Call Ollama chat model and return textual content (defensive)."""
    response = llm.invoke(prompt)
    # response.content expected; be defensive
    return getattr(response, "content", str(response))

def is_meta_query(query: str) -> bool:
    """Simple heuristic to detect meta/conversational queries."""
    q = (query or "").lower()
    meta_keywords = [
        "what did i just ask",
        "repeat my question",
        "what was my last question",
        "summarize our chat",
        "what did you just say",
        "repeat that",
        "what did i ask",
        "what did i just ask?"
    ]
    return any(keyword in q for keyword in meta_keywords)

def format_history(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    """Format the last N turns of chat history for the LLM prompt."""
    if not history:
        return ""
    # Keep last max_turns user+assistant turns (approx 2*max_turns messages)
    relevant = history[-(max_turns * 2):]
    parts = []
    for msg in relevant:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
