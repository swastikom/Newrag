import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Any, Tuple
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
import database as db
from contextlib import asynccontextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from server.functions.util import chunk_text, format_history, get_llm_response, get_text_from_pdf, is_meta_query
from server.models.chat_models import ChatRequest, ChatResponse, DocumentSummary

# Load environment variables
load_dotenv()

# --- Initialize Ollama Models (done before lifespan so they exist) ---
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="llama3.2")

# --- Application Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables and initialize Chroma vector store on startup."""
    print("Creating database tables...")
    db.create_tables()

    # Initialize Chroma vector store (global)
    global vector_store
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    yield

    print("Application shutdown complete.")

app = FastAPI(
    title="RAG Document Chat API",
    description="A FastAPI backend for a Retrieval-Augmented Generation (RAG) system.",
    version="0.2.0",
    lifespan=lifespan
)

# --- API Endpoints ---
@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, store full text in SQLite, chunk & add embeddings to Chroma.
    """
    try:
        # 1. Extract raw text
        text = get_text_from_pdf(file)

        # 2. Chunk the text for embeddings
        chunks = chunk_text(text)

        # 3. Generate document id
        doc_id = str(uuid.uuid4())

        # 4. Store metadata + full text in SQLite
        db.add_document(doc_id, file.filename, text)

        # 5. Store chunks + metadata in Chroma (if any chunks exist)
        if chunks:
            metadatas = [{"document_id": doc_id} for _ in chunks]
            vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas,
                ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
            )

        return JSONResponse(content={"message": "Document processed and stored successfully", "document_id": doc_id})
    except Exception as e:
        # Log and surface a friendly error
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/documents", response_model=List[DocumentSummary])
async def list_documents():
    """
    Returns list of documents with a short preview (first 200 chars safely).
    Defensive: handles content==None.
    """
    try:
        docs = db.get_documents()
        result = []
        for d in docs:
            content = d.get("content") if isinstance(d, dict) else None
            # fallback to empty string if None
            preview = (content or "")[:200] + ("..." if content and len(content) > 200 else "")
            result.append({
                "document_id": d.get("document_id"),
                "filename": d.get("filename"),
                "preview": preview
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/chats/{document_id}", response_model=List[ChatResponse])
async def get_chats(document_id: str):
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")
    return db.get_chats(document_id)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    document_id = request.document_id
    query = request.query or ""

    # Validate document exists
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")

    # Save user message to history immediately
    db.add_chat_message(document_id, "user", query)

    # 1) If meta query -> include history and ask LLM (bypass TF-IDF)
    if is_meta_query(query):
        history = db.get_chats(document_id)
        history_text = format_history(history, max_turns=5)
        prompt = (
            "You are a helpful assistant. Use the conversation history to answer the user's meta request. If user's main topic is not related to the current document, then let the user know else proceed with your answer.\n\n"
            f"Conversation History:\n{history_text}\n\nUser Meta Request:\n{query}\n"
        )
        llm_response = get_llm_response(prompt)
        db.add_chat_message(document_id, "model", llm_response)
        return ChatResponse(role="model", content=llm_response)

    # 2) TF-IDF similarity check against stored full text
    full_text = db.get_full_document_text(document_id) or ""
    if not full_text.strip():
        fallback_message = "The document has no content to analyze."
        db.add_chat_message(document_id, "model", fallback_message)
        return ChatResponse(role="model", content=fallback_message)

    try:
        vectorizer = TfidfVectorizer().fit([full_text, query])
        tfidf_matrix = vectorizer.transform([full_text, query])
        similarity_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        # in case TF-IDF fails (e.g., empty query), treat as 0 similarity
        similarity_score = 0.0

    # threshold - tweakable
    tfidf_threshold = 0.1
    if similarity_score < tfidf_threshold:
        # Consider request unrelated -> do not hit LLM
        fallback_message = "Your question does not seem related to this document. Please ask something relevant."
        db.add_chat_message(document_id, "model", fallback_message)
        return ChatResponse(role="model", content=fallback_message)

    # 3) Use Chroma embeddings search to get context chunks
    docs_and_scores: List[Tuple[Any, float]] = vector_store.similarity_search_with_score(
        query=query,
        k=3,
        filter={"document_id": document_id}
    )

    # filter by score threshold (embedding score semantics depend on vectorstore implementation)
    score_threshold = 0.90
    relevant_chunks = [doc.page_content for doc, score in docs_and_scores if score >= score_threshold]

    # If no chunk passes the threshold, be conservative and return fallback
    if not relevant_chunks:
        fallback_message = "I'm sorry, I cannot answer this question based on the content of this document."
        db.add_chat_message(document_id, "model", fallback_message)
        return ChatResponse(role="model", content=fallback_message)

    # 4) Build prompt that includes context + recent history
    history = db.get_chats(document_id)
    history_text = format_history(history, max_turns=5)
    context = "\n---\n".join(relevant_chunks)

    prompt = (
        "You are a helpful assistant. Answer using ONLY the document context and the conversation history.\n\n"
        f"Document Context:\n{context}\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User Question:\n{query}\n"
    )

    llm_response = get_llm_response(prompt)
    db.add_chat_message(document_id, "model", llm_response)

    return ChatResponse(role="model", content=llm_response)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        results = vector_store.get(where={"document_id": document_id})
        ids_to_delete = results.get('ids', [])
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
        db.delete_document_data(document_id)
        return JSONResponse(content={"message": "Document and all related data deleted successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
