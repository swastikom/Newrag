import os
import io
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
import database as db
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# --- Application Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    On startup, it creates the database tables and initializes the ChromaDB client.
    """
    print("Creating database tables...")
    db.create_tables()
    
    # Initialize ChromaDB client here
    global vector_store
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    yield
    print("Application shutdown complete.")

# Initialize FastAPI app with the lifespan event handler
app = FastAPI(
    title="RAG Document Chat API",
    description="A FastAPI backend for a Retrieval-Augmented Generation (RAG) system. It allows users to upload PDF documents, and chat with them. Data is stored in an SQLite database and ChromaDB.",
    version="0.1.0",
    lifespan=lifespan
)

# --- Initialize Ollama Models ---
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="llama3.2")

# --- Utility Functions ---

def get_text_from_pdf(file: UploadFile) -> str:
    """Extracts text from an uploaded PDF file."""
    pdf_reader = PdfReader(io.BytesIO(file.file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Splits text into chunks of a given size with some overlap."""
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))
        chunk = text[current_pos:end_pos]
        chunks.append(chunk)
        current_pos += chunk_size
    return chunks

def get_llm_response(prompt: str) -> str:
    """
    Calls the Ollama chat model to get a grounded response.
    """
    response = llm.invoke(prompt)
    return response.content

# --- API Models ---

class ChatRequest(BaseModel):
    document_id: str
    query: str

class ChatResponse(BaseModel):
    role: str
    content: str

class DocumentSummary(BaseModel):
    document_id: str
    filename: str

# --- API Endpoints ---

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF document, processes it, and stores its chunks and embeddings in ChromaDB.
    """
    try:
        # 1. Extract text from PDF
        text = get_text_from_pdf(file)
        
        # 2. Chunk the text
        chunks = chunk_text(text)
        
        # 3. Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # 4. Store document metadata in the SQLite database
        db.add_document(doc_id, file.filename)
        
        # 5. Add chunks and embeddings to ChromaDB
        # We need to add document IDs to the metadata to enable filtering
        metadata = [{"document_id": doc_id} for _ in chunks]
        vector_store.add_texts(
            texts=chunks,
            metadatas=metadata,
            ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
        )

        return JSONResponse(content={"message": "Document processed and stored successfully", "document_id": doc_id})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/documents", response_model=List[DocumentSummary])
async def list_documents():
    """Retrieves a list of all uploaded documents from the database."""
    return db.get_documents()

@app.get("/chats/{document_id}", response_model=List[ChatResponse])
async def get_chats(document_id: str):
    """Retrieves all chat messages for a specific document."""
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")
    
    return db.get_chats(document_id)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """
    Processes a user's query against a document, retrieves context from ChromaDB, and generates a response.
    """
    document_id = request.document_id
    query = request.query
    
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")
        
    # 1. Perform a similarity search, filtering by document_id
    relevant_docs: List[Any] = vector_store.similarity_search(
        query=query,
        k=3,
        filter={"document_id": document_id}
    )
    
    # Check for relevant context. If none is found, return a polite message
    if not relevant_docs:
        print("No relevant context found for the query. Sending fallback message.")
        fallback_message = (
            "I'm sorry, I cannot answer this question based on the content of this document. "
            "Please ask a question that is directly related to this document."
        )
        # Store the user's message and the fallback message
        db.add_chat_message(document_id, "user", query)
        db.add_chat_message(document_id, "model", fallback_message)
        
        return ChatResponse(role="model", content=fallback_message)
    
    # 2. Extract the context from the search results
    print(f"Found {len(relevant_docs)} relevant documents for the query.")
    relevant_chunks = [doc.page_content for doc in relevant_docs]
    print(f"Relevant Chunks: {relevant_chunks}")
    context = "\n---\n".join(relevant_chunks)
    
    # 3. Construct the RAG prompt for the LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    # 4. Call the LLM to get a grounded response
    llm_response = get_llm_response(prompt)
    
    # 5. Store the user's message and the LLM's response
    db.add_chat_message(document_id, "user", query)
    db.add_chat_message(document_id, "model", llm_response)
    
    return ChatResponse(role="model", content=llm_response)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Deletes a document and all associated chats and context from both databases."""
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")
        
    try:
        # Get the IDs of the documents to be deleted
        results = vector_store.get(where={"document_id": document_id})
        ids_to_delete = results.get('ids', [])

        # Delete data from ChromaDB by ID
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
        
        # Delete data from the SQLite database
        db.delete_document_data(document_id)
        
        return JSONResponse(content={"message": "Document and all related data deleted successfully"})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
