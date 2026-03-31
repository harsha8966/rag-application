"""
Embeddings Module

Handles vector embedding generation and storage:
- Gemini embedding generation
- FAISS vector store operations
- Index persistence and loading
"""

from app.core.embeddings.gemini_embeddings import GeminiEmbeddings
from app.core.embeddings.vector_store import FAISSVectorStore

__all__ = ["GeminiEmbeddings", "FAISSVectorStore"]
