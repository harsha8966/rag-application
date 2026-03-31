"""
Score Reranker - Post-Retrieval Quality Enhancement

This module provides reranking capabilities to improve retrieval quality
after the initial vector search.

Why Reranking?
- Vector similarity is fast but approximate
- Reranking can apply more sophisticated relevance signals
- Can incorporate business logic (boost recent docs, etc.)
- Score thresholding removes irrelevant results
"""

from typing import List, Optional
from dataclasses import dataclass

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


# Forward declaration for type hint
class RetrievedChunk:
    content: str
    score: float
    source_filename: str
    source_page: int
    chunk_id: str
    metadata: dict


class ScoreReranker:
    """
    Reranks retrieved chunks based on multiple signals.
    
    Current implementation uses score thresholding.
    Can be extended with:
    - Cross-encoder reranking
    - Business rule boosts
    - Recency weighting
    - User preference learning
    """
    
    def __init__(
        self,
        score_threshold: Optional[float] = None,
        boost_exact_match: bool = True
    ):
        """
        Initialize reranker.
        
        Args:
            score_threshold: Minimum score to keep
            boost_exact_match: Boost chunks with exact query term matches
        """
        settings = get_settings()
        self.score_threshold = score_threshold or settings.similarity_threshold
        self.boost_exact_match = boost_exact_match
    
    def rerank(
        self,
        chunks: List,  # List[RetrievedChunk]
        query: str,
        top_k: Optional[int] = None
    ) -> List:  # List[RetrievedChunk]
        """
        Rerank chunks based on multiple signals.
        
        Process:
        1. Filter by score threshold
        2. Apply query term boost
        3. Sort by adjusted score
        4. Return top-k
        
        Args:
            chunks: Initial retrieved chunks
            query: Original query for term matching
            top_k: Maximum chunks to return
            
        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return []
        
        # Calculate adjusted scores
        scored_chunks = []
        query_terms = set(query.lower().split())
        
        for chunk in chunks:
            # Start with original score
            adjusted_score = chunk.score
            
            # Apply score threshold
            if adjusted_score < self.score_threshold:
                continue
            
            # Boost exact term matches
            if self.boost_exact_match:
                content_lower = chunk.content.lower()
                matches = sum(1 for term in query_terms if term in content_lower)
                match_ratio = matches / len(query_terms) if query_terms else 0
                
                # Small boost for term presence (max 10%)
                adjusted_score += adjusted_score * match_ratio * 0.1
            
            scored_chunks.append((chunk, adjusted_score))
        
        # Sort by adjusted score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Extract chunks
        reranked = [chunk for chunk, _ in scored_chunks]
        
        # Apply top-k limit
        if top_k:
            reranked = reranked[:top_k]
        
        logger.debug(
            "Reranking completed",
            input_count=len(chunks),
            output_count=len(reranked)
        )
        
        return reranked
    
    def filter_by_agreement(
        self,
        chunks: List,  # List[RetrievedChunk]
        min_agreement: int = 2
    ) -> List:  # List[RetrievedChunk]
        """
        Filter to chunks that agree with at least N others.
        
        Useful for:
        - Reducing noise from outlier chunks
        - Increasing confidence in answers
        - Multi-document verification
        
        Args:
            chunks: Retrieved chunks
            min_agreement: Minimum chunks from same document
            
        Returns:
            Filtered chunks
        """
        if len(chunks) < min_agreement:
            return chunks
        
        # Count chunks per source
        source_counts: dict = {}
        for chunk in chunks:
            source = chunk.source_filename
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Filter to sources with enough chunks
        filtered = [
            chunk for chunk in chunks
            if source_counts[chunk.source_filename] >= min_agreement
        ]
        
        return filtered if filtered else chunks  # Fallback to original if all filtered
