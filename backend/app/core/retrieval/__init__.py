"""
Retrieval Module

Handles the retrieval strategy for finding relevant document chunks:
- Similarity-based retrieval
- MMR for diversity
- Score thresholding and reranking
"""

from app.core.retrieval.retriever import DocumentRetriever
from app.core.retrieval.reranker import ScoreReranker

__all__ = ["DocumentRetriever", "ScoreReranker"]
