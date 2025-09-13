import sqlite3
import json
import datetime
from typing import List, Dict, Any

from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import relationship

# The SQLAlchemy database URL you provided
SQLALCHEMY_DATABASE_URL = "sqlite:///./rag.db"

# Create a SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a declarative base to define tables
Base = declarative_base()

# --- SQLAlchemy Models ---

class Document(Base):
    __tablename__ = 'documents'

    document_id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    
    # Define a relationship to the chats table
    chats = relationship("Chat", back_populates="document", cascade="all, delete-orphan")

class Chat(Base):
    __tablename__ = 'chats'

    chat_id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey('documents.document_id'))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now)

    # Define a relationship back to the document table
    document = relationship("Document", back_populates="chats")

# Create a session to interact with the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Functions ---

def create_tables():
    """
    Creates the necessary tables for documents, chats, and context.
    """
    Base.metadata.create_all(bind=engine)

def add_document(document_id: str, filename: str):
    """Adds a new document to the documents table."""
    db = SessionLocal()
    try:
        new_doc = Document(document_id=document_id, filename=filename)
        db.add(new_doc)
        db.commit()
    finally:
        db.close()

def get_documents() -> List[Dict[str, Any]]:
    """Retrieves all documents from the database."""
    db = SessionLocal()
    try:
        documents = db.query(Document).all()
        return [{"document_id": doc.document_id, "filename": doc.filename} for doc in documents]
    finally:
        db.close()

def add_chat_message(document_id: str, role: str, content: str):
    """Adds a new chat message to the chats table."""
    db = SessionLocal()
    try:
        new_chat = Chat(document_id=document_id, role=role, content=content)
        db.add(new_chat)
        db.commit()
    finally:
        db.close()

def get_chats(document_id: str) -> List[Dict[str, Any]]:
    """Retrieves all chat messages for a specific document, ordered by timestamp."""
    db = SessionLocal()
    try:
        chats = db.query(Chat).filter(Chat.document_id == document_id).order_by(Chat.timestamp).all()
        return [{"role": chat.role, "content": chat.content} for chat in chats]
    finally:
        db.close()

def delete_document_data(document_id: str):
    """
    Deletes a document and all associated chats.
    """
    db = SessionLocal()
    try:
        document_to_delete = db.query(Document).filter(Document.document_id == document_id).first()
        if document_to_delete:
            db.delete(document_to_delete)
            db.commit()
    finally:
        db.close()
