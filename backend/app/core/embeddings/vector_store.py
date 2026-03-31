"""
FAISS Vector Store - Efficient Similarity Search

This module manages the FAISS vector index for storing and searching embeddings.

=== WHY FAISS ===

1. PERFORMANCE
   - Highly optimized C++ with Python bindings
   - Sub-millisecond search on millions of vectors
   - GPU support available for massive scale

2. LOCAL OPERATION
   - No external service dependencies
   - Data stays on-premises (enterprise requirement)
   - No network latency for searches

3. FLEXIBILITY
   - Multiple index types for different tradeoffs
   - Easy to serialize/deserialize
   - Works offline

4. COST
   - Free and open source
   - No per-query pricing
   - No vendor lock-in

=== INDEX TYPE CHOICE ===

Using IndexFlatIP (Inner Product with L2 normalization = Cosine Similarity):
- Exact search (no approximation)
- Best accuracy
- Fast enough for < 1M vectors
- For larger scale: consider IndexIVFFlat or IndexHNSW

=== METADATA STORAGE ===

FAISS only stores vectors, not metadata. We maintain a parallel
metadata store (JSON file) that maps vector indices to chunk metadata.
This enables returning source citations with search results.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import faiss

from app.core.embeddings.gemini_embeddings import GeminiEmbeddings, get_embeddings
from app.core.ingestion.chunker import DocumentChunk
from app.config import get_settings
from app.utils.exceptions import RetrievalError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store with metadata support.
    
    Features:
    - Add documents with automatic embedding
    - Similarity search with score filtering
    - MMR (Maximum Marginal Relevance) search
    - Persistent storage and loading
    - Metadata management for citations
    
    Usage:
        store = FAISSVectorStore()
        store.add_chunks(chunks)
        results = store.similarity_search("query", k=5)
        store.save()
    """
    
    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.json"
    
    def __init__(
        self,
        embeddings: Optional[GeminiEmbeddings] = None,
        index_path: Optional[Path] = None
    ):
        """
        Initialize vector store.
        
        Args:
            embeddings: Embedding generator. If None, uses default.
            index_path: Path for index persistence. If None, uses config.
        """
        settings = get_settings()
        
        self.embeddings = embeddings or get_embeddings()
        self.index_path = index_path or settings.faiss_index_directory
        self.index_path = Path(self.index_path)
        
        # Ensure directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load index
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict[str, Any]] = []
        self._document_ids: set = set()  # Track ingested documents
        
        # Try to load existing index
        if self._index_exists():
            self._load()
        else:
            self._create_new_index()
        
        logger.info(
            "FAISSVectorStore initialized",
            index_path=str(self.index_path),
            total_vectors=self._index.ntotal if self._index else 0
        )
    
    def _create_new_index(self) -> None:
        """Create a new empty FAISS index."""
        # Using IndexFlatIP for cosine similarity
        # (vectors are normalized before adding)
        self._index = faiss.IndexFlatIP(self.embeddings.dimension)
        self._metadata = []
        self._document_ids = set()
        
        logger.info(
            "Created new FAISS index",
            dimension=self.embeddings.dimension
        )
    
    def _index_exists(self) -> bool:
        """Check if a saved index exists."""
        index_file = self.index_path / self.INDEX_FILENAME
        metadata_file = self.index_path / self.METADATA_FILENAME
        return index_file.exists() and metadata_file.exists()
    
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        chunk_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Handles:
        - Embedding generation
        - Vector normalization (for cosine similarity)
        - Metadata storage
        - Automatic saving
        
        Args:
            chunks: List of DocumentChunk objects
            chunk_metadata: Optional pre-computed metadata for each chunk
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        logger.info("Adding chunks to vector store", count=len(chunks))
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Convert to numpy array and normalize for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        
        # Prepare metadata
        if chunk_metadata is None:
            chunk_metadata = [chunk.to_dict() for chunk in chunks]
        
        # Add to index
        self._index.add(vectors)
        self._metadata.extend(chunk_metadata)
        
        # Track document IDs
        for chunk in chunks:
            self._document_ids.add(chunk.source_filename)
        
        # Auto-save after adding
        self.save()
        
        logger.info(
            "Chunks added successfully",
            added=len(chunks),
            total=self._index.ntotal
        )
        
        return len(chunks)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (metadata, score) tuples, sorted by relevance
        """
        if self._index.ntotal == 0:
            logger.warning("Search attempted on empty index")
            return []
        
        settings = get_settings()
        score_threshold = score_threshold or settings.similarity_threshold
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search index
        scores, indices = self._index.search(query_vector, min(k, self._index.ntotal))
        
        # Build results with metadata
        results: List[Tuple[Dict[str, Any], float]] = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Convert score to 0-1 range (already normalized, so score is cosine)
            similarity = float(score)
            
            # Apply threshold
            if score_threshold and similarity < score_threshold:
                continue
            
            metadata = self._metadata[idx].copy()
            metadata["similarity_score"] = similarity
            
            results.append((metadata, similarity))
        
        logger.info(
            "Similarity search completed",
            query_length=len(query),
            results=len(results),
            top_score=results[0][1] if results else 0
        )
        
        return results
    
    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Maximum Marginal Relevance search for diverse results.
        
        MMR balances relevance and diversity:
        - High lambda_mult (→1): Favor relevance (similar to regular search)
        - Low lambda_mult (→0): Favor diversity (more varied results)
        
        Why MMR?
        - Avoid redundant results from similar chunks
        - Better coverage of different aspects of a topic
        - Improves multi-document reasoning
        
        Args:
            query: Search query text
            k: Number of results to return
            fetch_k: Number of candidates to consider
            lambda_mult: Diversity factor (0-1)
            
        Returns:
            List of (metadata, score) tuples
        """
        if self._index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Get more candidates than needed
        fetch_k = min(fetch_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, fetch_k)
        
        # Get candidate embeddings for MMR calculation
        candidate_indices = [i for i in indices[0] if i != -1]
        if not candidate_indices:
            return []
        
        # Reconstruct candidate vectors
        candidate_vectors = np.zeros((len(candidate_indices), self.embeddings.dimension), dtype=np.float32)
        for i, idx in enumerate(candidate_indices):
            self._index.reconstruct(int(idx), candidate_vectors[i])
        
        # MMR selection
        selected_indices: List[int] = []
        candidate_set = set(range(len(candidate_indices)))
        
        while len(selected_indices) < k and candidate_set:
            best_score = -float('inf')
            best_idx = -1
            
            for i in candidate_set:
                # Relevance to query
                relevance = float(np.dot(query_vector[0], candidate_vectors[i]))
                
                # Maximum similarity to already selected
                if selected_indices:
                    selected_vectors = candidate_vectors[selected_indices]
                    similarities = np.dot(selected_vectors, candidate_vectors[i])
                    max_sim = float(np.max(similarities))
                else:
                    max_sim = 0
                
                # MMR score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                candidate_set.remove(best_idx)
        
        # Build results
        results: List[Tuple[Dict[str, Any], float]] = []
        for i in selected_indices:
            original_idx = candidate_indices[i]
            score = float(scores[0][i])
            
            metadata = self._metadata[original_idx].copy()
            metadata["similarity_score"] = score
            
            results.append((metadata, score))
        
        logger.info(
            "MMR search completed",
            query_length=len(query),
            results=len(results),
            lambda_mult=lambda_mult
        )
        
        return results
    
    def save(self) -> None:
        """
        Persist index and metadata to disk.
        
        Saves:
        - FAISS index binary file
        - Metadata JSON file
        """
        index_file = self.index_path / self.INDEX_FILENAME
        metadata_file = self.index_path / self.METADATA_FILENAME
        
        # Save FAISS index
        faiss.write_index(self._index, str(index_file))
        
        # Save metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": self._metadata,
                "document_ids": list(self._document_ids)
            }, f, indent=2)
        
        logger.info(
            "Vector store saved",
            index_file=str(index_file),
            vectors=self._index.ntotal
        )
    
    def _load(self) -> None:
        """Load index and metadata from disk."""
        index_file = self.index_path / self.INDEX_FILENAME
        metadata_file = self.index_path / self.METADATA_FILENAME
        
        try:
            # Load FAISS index
            self._index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._metadata = data["metadata"]
                self._document_ids = set(data.get("document_ids", []))
            
            logger.info(
                "Vector store loaded",
                vectors=self._index.ntotal,
                documents=len(self._document_ids)
            )
            
        except Exception as e:
            logger.error("Failed to load vector store", error=str(e))
            raise RetrievalError(
                f"Failed to load vector store: {str(e)}",
                details={"index_path": str(self.index_path)}
            )
    
    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks from a specific document.
        
        Note: FAISS doesn't support deletion, so we rebuild the index
        without the deleted document's vectors.
        
        Args:
            document_id: Document filename to delete
            
        Returns:
            Number of chunks deleted
        """
        if document_id not in self._document_ids:
            return 0
        
        # Find indices to keep
        keep_indices = []
        for i, meta in enumerate(self._metadata):
            if meta.get("source_filename") != document_id:
                keep_indices.append(i)
        
        deleted_count = len(self._metadata) - len(keep_indices)
        
        if deleted_count == 0:
            return 0
        
        # Reconstruct vectors to keep
        if keep_indices:
            vectors = np.zeros((len(keep_indices), self.embeddings.dimension), dtype=np.float32)
            for i, idx in enumerate(keep_indices):
                self._index.reconstruct(idx, vectors[i])
            
            new_metadata = [self._metadata[i] for i in keep_indices]
        else:
            vectors = np.array([], dtype=np.float32).reshape(0, self.embeddings.dimension)
            new_metadata = []
        
        # Rebuild index
        self._create_new_index()
        
        if len(vectors) > 0:
            self._index.add(vectors)
            self._metadata = new_metadata
            self._document_ids = {m["source_filename"] for m in new_metadata}
        
        self.save()
        
        logger.info(
            "Document deleted from vector store",
            document_id=document_id,
            deleted_chunks=deleted_count
        )
        
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "total_documents": len(self._document_ids),
            "document_ids": list(self._document_ids),
            "dimension": self.embeddings.dimension,
            "index_path": str(self.index_path)
        }
    
    def is_empty(self) -> bool:
        """Check if the vector store is empty."""
        return self._index is None or self._index.ntotal == 0
