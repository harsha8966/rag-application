"""
Gemini Embeddings - Vector Generation using Google's Embedding Model

This module handles the conversion of text into dense vector representations
using Google's Gemini text-embedding model.

=== WHY VECTOR SEARCH INSTEAD OF KEYWORD SEARCH ===

1. SEMANTIC UNDERSTANDING
   - Keyword search: "car" won't match "automobile" or "vehicle"
   - Vector search: Understands semantic similarity
   - Example: Query "How to fix engine problems" matches content about
     "troubleshooting motor issues" even without shared keywords

2. CONTEXT-AWARE MATCHING
   - Keywords ignore context: "bank" matches both financial and river
   - Vectors capture meaning in context
   - Better precision for ambiguous queries

3. HANDLING PARAPHRASING
   - Users don't know exact document wording
   - Vector search finds conceptually similar content
   - Critical for natural language queries

4. MULTILINGUAL POTENTIAL
   - Same concept in different languages = similar vectors
   - Enables cross-lingual retrieval (if model supports it)

=== WHY GEMINI EMBEDDINGS ===

1. HIGH QUALITY
   - State-of-the-art embedding model
   - 768-dimensional vectors capture rich semantics
   - Trained on diverse, high-quality data

2. GENEROUS LIMITS
   - Higher rate limits than some alternatives
   - Cost-effective for enterprise use
   - No self-hosting required

3. CONSISTENCY
   - Using same provider for embeddings and LLM
   - Potentially better alignment between retrieval and generation
   - Simplified API key management
"""

from typing import List, Optional
import asyncio
import time
from functools import lru_cache

import google.generativeai as genai

from app.config import get_settings
from app.utils.exceptions import EmbeddingError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiEmbeddings:
    """
    Generate embeddings using Google's Gemini text-embedding model.
    
    Features:
    - Batch embedding for efficiency
    - Automatic retry with exponential backoff
    - Caching for repeated queries
    - Async support for high throughput
    
    Usage:
        embedder = GeminiEmbeddings()
        vectors = embedder.embed_documents(["text1", "text2"])
        query_vector = embedder.embed_query("search query")
    """
    
    # gemini-embedding-001 produces 3072-dimensional vectors
    # (older text-embedding-004 used 768)
    EMBEDDING_DIMENSION = 3072
    
    # Maximum texts per batch (API limit)
    MAX_BATCH_SIZE = 100
    
    # Delay between batches in seconds to avoid API rate limits
    BATCH_DELAY_SECONDS = 1.5
    
    # Retry settings for transient API failures
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini embeddings client.
        
        Args:
            api_key: Google API key. If None, loaded from config.
            model: Embedding model name. If None, loaded from config.
        """
        settings = get_settings()
        
        self.api_key = api_key or settings.google_api_key
        self.model = model or settings.embedding_model
        
        if not self.api_key:
            raise EmbeddingError(
                "Google API key not configured",
                details={"hint": "Set GOOGLE_API_KEY in .env file"}
            )
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        logger.info(
            "GeminiEmbeddings initialized",
            model=self.model,
            dimension=self.EMBEDDING_DIMENSION
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Handles batching automatically for large lists.
        Uses task_type="retrieval_document" for document embeddings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (768-dimensional each)
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []
        
        logger.info("Generating document embeddings", count=len(texts))
        
        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE
        
        # Process in batches with delay to avoid rate limits
        for batch_num, i in enumerate(range(0, len(texts), self.MAX_BATCH_SIZE), start=1):
            batch = texts[i:i + self.MAX_BATCH_SIZE]
            
            logger.info(
                "Processing embedding batch",
                batch=batch_num,
                total_batches=total_batches,
                batch_size=len(batch)
            )
            
            batch_embeddings = self._embed_batch(
                batch, 
                task_type="retrieval_document"
            )
            all_embeddings.extend(batch_embeddings)
            
            # Delay between batches to respect API rate limits
            if batch_num < total_batches:
                time.sleep(self.BATCH_DELAY_SECONDS)
        
        logger.info(
            "Document embeddings generated",
            count=len(all_embeddings),
            dimension=self.EMBEDDING_DIMENSION
        )
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Uses task_type="retrieval_query" which is optimized for queries.
        This subtle difference improves retrieval quality.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector (768-dimensional)
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        logger.debug("Generating query embedding", text_length=len(text))
        
        embeddings = self._embed_batch([text], task_type="retrieval_query")
        
        return embeddings[0]
    
    def _embed_batch(
        self, 
        texts: List[str], 
        task_type: str = "retrieval_document"
    ) -> List[List[float]]:
        """
        Internal method to embed a batch of texts with retry logic.
        
        Retries with exponential backoff on transient failures
        (rate limits, server errors).
        
        Args:
            texts: Batch of texts to embed
            task_type: Either "retrieval_document" or "retrieval_query"
            
        Returns:
            List of embedding vectors
        """
        # Clean texts - remove empty strings and very long texts
        cleaned_texts = []
        for text in texts:
            if len(text) > 10000:
                text = text[:10000]
            cleaned_texts.append(text if text.strip() else " ")
        
        last_exception = None
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=cleaned_texts,
                    task_type=task_type
                )
                
                embeddings = result["embedding"]
                
                # Handle single vs batch response
                if cleaned_texts and not isinstance(embeddings[0], list):
                    embeddings = [embeddings]
                
                return embeddings
                
            except Exception as e:
                last_exception = e
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Embedding batch failed, retrying",
                        attempt=attempt,
                        max_retries=self.MAX_RETRIES,
                        retry_delay=delay,
                        error=str(e)
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Embedding generation failed after all retries",
                        error=str(e),
                        batch_size=len(texts),
                        attempts=self.MAX_RETRIES
                    )
        
        raise EmbeddingError(
            f"Failed to generate embeddings after {self.MAX_RETRIES} attempts: {str(last_exception)}",
            details={"batch_size": len(texts), "error": str(last_exception)}
        )
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of embed_documents.
        
        Useful for high-throughput scenarios where multiple
        embedding requests can be processed concurrently.
        """
        # Run sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension for FAISS index initialization."""
        return self.EMBEDDING_DIMENSION


@lru_cache(maxsize=1)
def get_embeddings() -> GeminiEmbeddings:
    """
    Get cached embeddings instance.
    
    Singleton pattern ensures we reuse the same client,
    avoiding repeated API configuration.
    """
    return GeminiEmbeddings()
