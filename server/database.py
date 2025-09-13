import datetime
from typing import List, Dict, Any
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# SQLAlchemy database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./rag.db"

# Engine & base
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Models ---
class Document(Base):
    __tablename__ = 'documents'
    document_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content = Column(Text, nullable=True)  # store full PDF text
    # relationship with chats
    chats = relationship("Chat", back_populates="document", cascade="all, delete-orphan")

class Chat(Base):
    __tablename__ = 'chats'
    chat_id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey('documents.document_id'))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    document = relationship("Document", back_populates="chats")

# --- Database helper functions ---
def create_tables():
    Base.metadata.create_all(bind=engine)

def add_document(document_id: str, filename: str, content: str):
    """Add a new document (including full text content)."""
    db = SessionLocal()
    try:
        new_doc = Document(document_id=document_id, filename=filename, content=content)
        db.add(new_doc)
        db.commit()
    finally:
        db.close()

def get_documents() -> List[Dict[str, Any]]:
    """Return all documents with content included (content may be None)."""
    db = SessionLocal()
    try:
        documents = db.query(Document).all()
        return [
            {"document_id": doc.document_id, "filename": doc.filename, "content": doc.content}
            for doc in documents
        ]
    finally:
        db.close()

def get_full_document_text(document_id: str) -> str:
    """Return the full text content for a document (or empty string)."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.document_id == document_id).first()
        return doc.content if doc and doc.content else ""
    finally:
        db.close()

def add_chat_message(document_id: str, role: str, content: str):
    """Add a chat message (user or model)."""
    db = SessionLocal()
    try:
        new_chat = Chat(document_id=document_id, role=role, content=content)
        db.add(new_chat)
        db.commit()
    finally:
        db.close()

def get_chats(document_id: str) -> List[Dict[str, Any]]:
    """Retrieve all chat messages for a document ordered by timestamp."""
    db = SessionLocal()
    try:
        chats = db.query(Chat).filter(Chat.document_id == document_id).order_by(Chat.timestamp).all()
        return [{"role": chat.role, "content": chat.content} for chat in chats]
    finally:
        db.close()

def delete_document_data(document_id: str):
    """Delete a document and all associated chats."""
    db = SessionLocal()
    try:
        document_to_delete = db.query(Document).filter(Document.document_id == document_id).first()
        if document_to_delete:
            db.delete(document_to_delete)
            db.commit()
    finally:
        db.close()
