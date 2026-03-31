"""
Document Retriever - Unified Retrieval Interface

This module provides a high-level interface for document retrieval,
combining various retrieval strategies.

=== HOW RETRIEVAL REDUCES HALLUCINATIONS ===

1. GROUNDING IN EVIDENCE
   - Without retrieval: LLM uses only its training data (may be outdated/wrong)
   - With retrieval: LLM answers ONLY from provided context
   - If relevant context isn't found → LLM can say "I don't know"

2. TRACEABLE ANSWERS
   - Every answer has specific source documents
   - Users can verify claims
   - Builds trust in the system

3. DOMAIN SPECIFICITY
   - Retrieved content is from YOUR documents
   - Not generic internet knowledge
   - Accurate for your specific domain/company

4. CONTROLLED INFORMATION
   - You decide what documents are searchable
   - Can update/remove outdated information
   - No risk of training data contamination

=== HOW MMR IMPROVES MULTI-DOCUMENT REASONING ===

Problem with pure similarity search:
- Top results may all be from same section
- Redundant information wastes context space
- Misses related info from other documents

MMR (Maximum Marginal Relevance) solution:
- Balances relevance AND diversity
- Second result is chosen to be both relevant and different from first
- Results cover more aspects of the question
- Better for questions that span multiple documents
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.embeddings.vector_store import FAISSVectorStore
from app.core.retrieval.reranker import ScoreReranker
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """
    A retrieved chunk with all necessary information for RAG.
    
    Contains:
    - The actual text content
    - Relevance score
    - Source information for citation
    """
    content: str
    score: float
    source_filename: str
    source_page: int
    chunk_id: str
    metadata: Dict[str, Any]
    
    def to_context_string(self) -> str:
        """Format chunk for inclusion in LLM context."""
        return (
            f"[Source: {self.source_filename}, Page {self.source_page}]\n"
            f"{self.content}"
        )
    
    def to_citation(self) -> Dict[str, Any]:
        """Format chunk as a citation for the frontend."""
        return {
            "source": self.source_filename,
            "page": self.source_page,
            "score": round(self.score, 3),
            "preview": self.content[:200] + "..." if len(self.content) > 200 else self.content
        }


class DocumentRetriever:
    """
    High-level document retrieval with multiple strategies.
    
    Supports:
    - Pure similarity search (fast, simple)
    - MMR search (diverse results)
    - Hybrid approaches
    - Score filtering
    
    Usage:
        retriever = DocumentRetriever(vector_store)
        chunks = retriever.retrieve("What is the refund policy?", k=5)
        
        # With MMR for diversity
        chunks = retriever.retrieve_mmr("Compare product features", k=5)
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        reranker: Optional[ScoreReranker] = None
    ):
        """
        Initialize retriever with vector store.
        
        Args:
            vector_store: FAISS vector store instance
            reranker: Optional reranker for post-processing
        """
        self.vector_store = vector_store
        self.reranker = reranker or ScoreReranker()
        
        settings = get_settings()
        self.default_k = settings.top_k_results
        self.score_threshold = settings.similarity_threshold
        self.mmr_lambda = settings.mmr_diversity_score
        
        logger.info(
            "DocumentRetriever initialized",
            default_k=self.default_k,
            threshold=self.score_threshold
        )
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_reranking: bool = True
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks using similarity search.
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            score_threshold: Minimum relevance score
            use_reranking: Whether to apply reranking
            
        Returns:
            List of RetrievedChunk objects
        """
        k = k or self.default_k
        score_threshold = score_threshold or self.score_threshold
        
        if self.vector_store.is_empty():
            logger.warning("Retrieval attempted on empty vector store")
            return []
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query=query,
            k=k * 2 if use_reranking else k,  # Fetch more for reranking
            score_threshold=score_threshold
        )
        
        # Convert to RetrievedChunk objects
        chunks = self._results_to_chunks(results)
        
        # Apply reranking if enabled
        if use_reranking and chunks:
            chunks = self.reranker.rerank(chunks, query)[:k]
        
        logger.info(
            "Retrieval completed",
            query_preview=query[:50],
            results=len(chunks)
        )
        
        return chunks
    
    def retrieve_mmr(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve with Maximum Marginal Relevance for diverse results.
        
        Best for:
        - Questions that span multiple topics
        - Comparative questions
        - When you want varied perspectives
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            fetch_k: Candidates to consider for MMR
            lambda_mult: Diversity factor (0=diverse, 1=relevant)
            
        Returns:
            List of RetrievedChunk objects
        """
        k = k or self.default_k
        lambda_mult = lambda_mult or self.mmr_lambda
        
        if self.vector_store.is_empty():
            return []
        
        # Perform MMR search
        results = self.vector_store.mmr_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        
        chunks = self._results_to_chunks(results)
        
        logger.info(
            "MMR retrieval completed",
            query_preview=query[:50],
            results=len(chunks),
            lambda_mult=lambda_mult
        )
        
        return chunks
    
    def retrieve_with_filters(
        self,
        query: str,
        k: Optional[int] = None,
        source_filter: Optional[List[str]] = None,
        page_filter: Optional[Tuple[int, int]] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve with metadata filters.
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            source_filter: Only include chunks from these files
            page_filter: Only include chunks from pages in range (start, end)
            
        Returns:
            Filtered list of RetrievedChunk objects
        """
        # Get more results than needed for filtering
        k = k or self.default_k
        chunks = self.retrieve(query, k=k * 3, use_reranking=False)
        
        # Apply filters
        filtered = []
        for chunk in chunks:
            # Source filter
            if source_filter and chunk.source_filename not in source_filter:
                continue
            
            # Page filter
            if page_filter:
                start, end = page_filter
                if not (start <= chunk.source_page <= end):
                    continue
            
            filtered.append(chunk)
            
            if len(filtered) >= k:
                break
        
        return filtered
    
    def _results_to_chunks(
        self,
        results: List[Tuple[Dict[str, Any], float]]
    ) -> List[RetrievedChunk]:
        """Convert raw search results to RetrievedChunk objects."""
        chunks = []
        
        for metadata, score in results:
            chunk = RetrievedChunk(
                content=metadata.get("content", ""),
                score=score,
                source_filename=metadata.get("source_filename", "Unknown"),
                source_page=metadata.get("source_page", 1),
                chunk_id=metadata.get("chunk_id", ""),
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_context_for_llm(
        self,
        chunks: List[RetrievedChunk],
        max_context_length: int = 8000
    ) -> str:
        """
        Format retrieved chunks as context for LLM.
        
        Args:
            chunks: Retrieved chunks
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the documents."
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[Document {i}: {chunk.source_filename}, Page {chunk.source_page}]\n{chunk.content}\n"
            
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n---\n".join(context_parts)
