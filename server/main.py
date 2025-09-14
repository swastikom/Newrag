import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Any, Tuple
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
import database as db
from contextlib import asynccontextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from functions.util import (
    chunk_text, format_history, get_llm_response,
    get_text_from_pdf, is_meta_query
)
from models.chat_models import ChatRequest, ChatResponse, DocumentSummary

# Load environment variables
load_dotenv()

# --- Initialize Ollama Models ---
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="llama3.2")

# --- Global Vector Store ---
vector_store: Chroma = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: create DB tables & restore vector store if empty."""
    print("Creating database tables...")
    db.create_tables()

    global vector_store
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    # 🟢 If chroma_db was deleted, rebuild embeddings from SQLite
    if not os.path.exists("./chroma_db") or not vector_store.get()["ids"]:
        print("⚠️ Chroma store empty. Rebuilding from DB...")
        docs = db.get_documents()
        for d in docs:
            text = d.get("content") or ""
            if not text.strip():
                continue
            chunks = chunk_text(text)
            metadatas = [{"document_id": d["document_id"]} for _ in chunks]
            ids = [f"{d['document_id']}-{i}" for i in range(len(chunks))]
            if chunks:
                vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
        print("✅ Rebuild complete.")

    yield
    print("Application shutdown complete.")


app = FastAPI(
    title="RAG Document Chat API",
    description="A FastAPI backend for a Retrieval-Augmented Generation (RAG) system.",
    version="0.3.0",
    lifespan=lifespan
)


# --- API Endpoints ---
@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, store text in SQLite, chunk & add embeddings to Chroma."""
    try:
        text = get_text_from_pdf(file)
        chunks = chunk_text(text)
        doc_id = str(uuid.uuid4())

        # Save in DB
        db.add_document(doc_id, file.filename, text)

        # Save in vector store
        if chunks:
            metadatas = [{"document_id": doc_id} for _ in chunks]
            vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas,
                ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
            )

        return JSONResponse(content={
            "message": "Document processed and stored successfully",
            "document_id": doc_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.get("/documents", response_model=List[DocumentSummary])
async def list_documents():
    """Return list of documents with short preview."""
    try:
        docs = db.get_documents()
        result = []
        for d in docs:
            content = d.get("content") if isinstance(d, dict) else None
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
    """Get all chats for a given document."""
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")
    return db.get_chats(document_id)


@app.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """Chat with a specific document."""
    document_id = request.document_id
    query = request.query or ""

    # Validate document exists
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")

    # Save user query
    db.add_chat_message(document_id, "user", query)

    # --- Meta query ---
    if is_meta_query(query):
        print("Meta query detected.")
        history = db.get_chats(document_id)
        history_text = format_history(history, max_turns=5)
        prompt = (
            "You are a helpful assistant. Use the conversation history to answer the user's meta request.\n\n"
            f"Conversation History:\n{history_text}\n\nUser Meta Request:\n{query}\n"
        )
        print("Hitting LLM for meta query...")
        llm_response = get_llm_response(prompt)
        db.add_chat_message(document_id, "model", llm_response)
        return ChatResponse(role="model", content=llm_response)
    print("No meta query. Normal query processing.")

    # --- TF-IDF check for relevance ---
    full_text = db.get_full_document_text(document_id) or ""
    if not full_text.strip():
        msg = "The document has no content to analyze."
        db.add_chat_message(document_id, "model", msg)
        return ChatResponse(role="model", content=msg)

    try:
        vectorizer = TfidfVectorizer().fit([full_text, query])
        tfidf_matrix = vectorizer.transform([full_text, query])
        similarity_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        similarity_score = 0.0

    if similarity_score < 0.08:  # lowered threshold
        print("Question-document similarity too low:", similarity_score)
        msg = "Your question does not seem related to this document."
        db.add_chat_message(document_id, "model", msg)
        print("Not hitting LLM.")
        return ChatResponse(role="model", content=msg)
    print("Question-document similarity greter than threshold:", similarity_score)

    # --- Embedding search ---
    docs_and_scores: List[Tuple[Any, float]] = vector_store.similarity_search_with_score(
        query=query,
        k=3,
        filter={"document_id": document_id}
    )

    if not docs_and_scores:
        print("No relevant chunks found in vector store.")
        msg = "No relevant chunks found in vector store."
        db.add_chat_message(document_id, "model", msg)
        return ChatResponse(role="model", content=msg)

    # Chroma returns (doc, score) with higher score = better similarity
    score_threshold = 0.5
    relevant_chunks = [doc.page_content for doc, score in docs_and_scores if score >= score_threshold]

    if not relevant_chunks:
        print("No chunks passed the relevance threshold.")
        print("Not hitting LLM.")
        msg = "I'm sorry, I cannot answer this question based on this document."
        db.add_chat_message(document_id, "model", msg)
        return ChatResponse(role="model", content=msg)

    # --- Build prompt ---
    print(f"Found {len(relevant_chunks)} relevant chunks. Building prompt...")
    history = db.get_chats(document_id)
    history_text = format_history(history, max_turns=5)
    context = "\n---\n".join(relevant_chunks)

    prompt = (
        "You are a helpful assistant. Answer using ONLY the document context and conversation history.\n\n"
        f"Document Context:\n{context}\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User Question:\n{query}\n"
    )

    print("Hitting LLM for answer...")
    llm_response = get_llm_response(prompt)
    db.add_chat_message(document_id, "model", llm_response)
    return ChatResponse(role="model", content=llm_response)


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings."""
    documents = db.get_documents()
    if not any(doc['document_id'] == document_id for doc in documents):
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        results = vector_store.get(where={"document_id": document_id})
        ids_to_delete = results.get('ids', [])
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
        db.delete_document_data(document_id)
        return JSONResponse(content={"message": "Document and related data deleted successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.post("/rebuild_index")
async def rebuild_index():
    """
    Rebuild Chroma vector store from documents stored in SQLite.
    Useful if ./chroma_db is deleted or corrupted.
    """
    global vector_store
    try:
        # Clear current vector store
        existing = vector_store.get()
        if existing and "ids" in existing and existing["ids"]:
            print(f"🗑️ Clearing {len(existing['ids'])} existing embeddings...")
            vector_store.delete(ids=existing["ids"])

        # Rebuild from DB
        docs = db.get_documents()
        total_chunks = 0
        for d in docs:
            text = d.get("content") or ""
            if not text.strip():
                continue
            chunks = chunk_text(text)
            metadatas = [{"document_id": d["document_id"]} for _ in chunks]
            ids = [f"{d['document_id']}-{i}" for i in range(len(chunks))]
            if chunks:
                vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
                total_chunks += len(chunks)

        return JSONResponse(content={
            "message": f"✅ Rebuilt Chroma index from {len(docs)} documents",
            "chunks_indexed": total_chunks
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")
