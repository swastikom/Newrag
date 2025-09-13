# --- API Models ---
from pydantic import BaseModel


class ChatRequest(BaseModel):
    document_id: str
    query: str

class ChatResponse(BaseModel):
    role: str
    content: str

class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    preview: str