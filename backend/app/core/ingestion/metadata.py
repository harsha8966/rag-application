"""
Metadata Extraction and Management

This module handles extraction and enrichment of document metadata.
Metadata is crucial for:
- Source citations in answers
- Filtering and faceted search
- Debugging and auditing
- Analytics and reporting

Metadata stored with each chunk:
- Source file information
- Page numbers
- Chunk position
- Processing timestamps
- Document statistics
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import hashlib

from app.core.ingestion.parser import ParsedDocument
from app.core.ingestion.chunker import DocumentChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentMetadata:
    """
    Complete metadata for an ingested document.
    
    This metadata is stored alongside the FAISS index
    for retrieval and citation purposes.
    """
    # Core identification
    document_id: str
    filename: str
    file_type: str
    
    # Content statistics
    total_pages: int
    total_chars: int
    total_chunks: int
    
    # Processing information
    ingested_at: str
    chunk_size_used: int
    chunk_overlap_used: int
    
    # Content hash for deduplication
    content_hash: str
    
    # Custom metadata (user-provided)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "total_chars": self.total_chars,
            "total_chunks": self.total_chunks,
            "ingested_at": self.ingested_at,
            "chunk_size_used": self.chunk_size_used,
            "chunk_overlap_used": self.chunk_overlap_used,
            "content_hash": self.content_hash,
            "custom_metadata": self.custom_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create from dictionary."""
        return cls(**data)


class MetadataExtractor:
    """
    Extracts and manages metadata for documents and chunks.
    
    Responsibilities:
    1. Generate unique document IDs
    2. Compute content hashes for deduplication
    3. Enrich chunks with metadata
    4. Track processing parameters
    """
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialize with chunking parameters for metadata tracking.
        
        Args:
            chunk_size: Chunk size used in processing
            chunk_overlap: Chunk overlap used in processing
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_document_metadata(
        self,
        document: ParsedDocument,
        chunks: List[DocumentChunk],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentMetadata:
        """
        Extract complete metadata for a processed document.
        
        Args:
            document: Parsed document
            chunks: Generated chunks
            custom_metadata: Optional user-provided metadata
            
        Returns:
            DocumentMetadata object
        """
        # Generate content hash for deduplication
        content_hash = self._compute_content_hash(document.full_text)
        
        # Generate unique document ID
        document_id = self._generate_document_id(document.filename, content_hash)
        
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=document.filename,
            file_type=document.file_type,
            total_pages=document.total_pages,
            total_chars=document.total_chars,
            total_chunks=len(chunks),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            chunk_size_used=self.chunk_size,
            chunk_overlap_used=self.chunk_overlap,
            content_hash=content_hash,
            custom_metadata=custom_metadata or {}
        )
        
        logger.info(
            "Document metadata extracted",
            document_id=document_id,
            filename=document.filename,
            chunks=len(chunks)
        )
        
        return metadata
    
    def enrich_chunk_metadata(
        self,
        chunk: DocumentChunk,
        document_metadata: DocumentMetadata
    ) -> Dict[str, Any]:
        """
        Create enriched metadata dictionary for a chunk.
        
        This metadata is stored in FAISS alongside the embedding
        and returned with search results for citation.
        
        Args:
            chunk: The document chunk
            document_metadata: Parent document metadata
            
        Returns:
            Dictionary of metadata for storage
        """
        return {
            # Chunk-specific
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
            
            # Source tracking (for citations)
            "source_filename": chunk.source_filename,
            "source_page": chunk.source_page,
            
            # Document-level context
            "document_id": document_metadata.document_id,
            "total_pages": document_metadata.total_pages,
            "file_type": document_metadata.file_type,
            
            # Processing info
            "ingested_at": document_metadata.ingested_at,
            
            # Custom metadata inherited from document
            **document_metadata.custom_metadata
        }
    
    def _compute_content_hash(self, content: str) -> str:
        """
        Compute SHA-256 hash of content for deduplication.
        
        Why hash?
        - Detect duplicate uploads
        - Track content changes
        - Efficient comparison
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    
    def _generate_document_id(self, filename: str, content_hash: str) -> str:
        """
        Generate unique document ID.
        
        Format: {sanitized_filename}_{content_hash}
        This ensures:
        - Same file with different content = different ID
        - Same content with different filename = different ID
        - Readable IDs for debugging
        """
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() else "_" for c in filename)
        safe_name = safe_name[:50]  # Limit length
        
        return f"{safe_name}_{content_hash}"
    
    def format_citation(
        self,
        chunk_metadata: Dict[str, Any],
        include_page: bool = True
    ) -> str:
        """
        Format chunk metadata as a citation string.
        
        Used when displaying sources to users.
        
        Args:
            chunk_metadata: Metadata from a retrieved chunk
            include_page: Whether to include page number
            
        Returns:
            Formatted citation string
        """
        filename = chunk_metadata.get("source_filename", "Unknown")
        
        if include_page:
            page = chunk_metadata.get("source_page", "?")
            return f"{filename}, Page {page}"
        
        return filename
