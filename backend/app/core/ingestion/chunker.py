"""
Document Chunker - Intelligent Text Splitting for RAG

This module splits documents into optimal chunks for embedding and retrieval.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.

=== WHY CHUNKING IS REQUIRED ===

1. EMBEDDING MODEL LIMITS
   - Embedding models have token limits (typically 512-8192 tokens)
   - Entire documents often exceed these limits
   - Chunking ensures each piece fits in the embedding window

2. RETRIEVAL PRECISION
   - Large documents contain mixed topics
   - Chunking allows retrieving only relevant sections
   - Improves answer accuracy by focusing context

3. CONTEXT WINDOW MANAGEMENT
   - LLMs have limited context windows
   - Multiple small chunks can be combined strategically
   - Better than truncating a large document

=== WHY OVERLAP IS REQUIRED ===

1. CONTEXT CONTINUITY
   - Information often spans chunk boundaries
   - Overlap ensures no information is "lost" at boundaries
   - Example: "The company was founded in 1995. It has grown..." 
     Without overlap, "It" might lose its referent

2. BETTER RETRIEVAL
   - Query might match text that spans two chunks
   - Overlap increases chance of matching relevant content
   - Improves recall without significantly impacting precision

3. SEMANTIC COHERENCE
   - Sentences often depend on previous context
   - Overlap preserves some of that context
   - Results in more meaningful embeddings

=== IMPACT ON RETRIEVAL QUALITY ===

Chunk Size:
- TOO SMALL (< 200 chars): Loses context, fragments meaning
- TOO LARGE (> 1500 chars): Dilutes relevance, wastes context window
- OPTIMAL (700-800 tokens ≈ 2500-3000 chars): Balances context and precision

Overlap:
- TOO SMALL (< 50 chars): Boundary information loss
- TOO LARGE (> 300 chars): Redundancy, storage waste
- OPTIMAL (100-150 tokens ≈ 400-600 chars): Good context preservation
"""

from typing import List, Optional
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.ingestion.parser import ParsedDocument, ParsedPage
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """
    A single chunk of text with full provenance metadata.
    
    Metadata is critical for:
    - Citation generation (show users where answer came from)
    - Debugging retrieval issues
    - Filtering by source if needed
    """
    chunk_id: str
    content: str
    char_count: int
    
    # Source tracking
    source_filename: str
    source_page: int
    chunk_index: int  # Position within document
    
    # Optional metadata
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage/serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "char_count": self.char_count,
            "source_filename": self.source_filename,
            "source_page": self.source_page,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata
        }


class DocumentChunker:
    """
    Splits documents into optimally-sized chunks for RAG.
    
    Uses RecursiveCharacterTextSplitter which:
    - Tries to split on paragraph boundaries first
    - Falls back to sentences, then words
    - Preserves semantic units as much as possible
    
    Configuration:
    - chunk_size: Target size in characters (~750 for 700-800 tokens)
    - chunk_overlap: Overlap between chunks (~100 tokens)
    
    Why RecursiveCharacterTextSplitter over others?
    - Respects document structure (paragraphs, sentences)
    - Configurable separators for different document types
    - Battle-tested in production RAG systems
    """
    
    # Separators in order of preference
    # Try to split on larger semantic units first
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks (highest priority)
        "\n",    # Line breaks
        ". ",    # Sentence ends
        "! ",    # Exclamation
        "? ",    # Questions
        "; ",    # Semicolons
        ", ",    # Commas
        " ",     # Words (last resort)
        ""       # Characters (emergency fallback)
    ]
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize chunker with configurable parameters.
        
        Args:
            chunk_size: Target chunk size in characters. 
                       Default from config (~750 chars ≈ 700-800 tokens)
            chunk_overlap: Overlap between chunks.
                          Default from config (~400 chars ≈ 100-150 tokens)
            separators: Custom separators for splitting.
                       Default uses semantic boundaries.
        """
        settings = get_settings()
        
        # Use provided values or fall back to config
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        
        # Initialize LangChain splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,  # Use character count
            is_separator_regex=False
        )
        
        logger.info(
            "DocumentChunker initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def chunk_document(self, document: ParsedDocument) -> List[DocumentChunk]:
        """
        Split a parsed document into chunks.
        
        Strategy:
        1. Process each page separately to maintain page attribution
        2. Split page content into chunks
        3. Assign unique IDs and metadata to each chunk
        
        Args:
            document: Parsed document from DocumentParser
            
        Returns:
            List of DocumentChunk objects ready for embedding
        """
        chunks: List[DocumentChunk] = []
        chunk_index = 0
        
        for page in document.pages:
            # Skip empty pages
            if not page.content.strip():
                continue
            
            # Split this page's content
            page_chunks = self._splitter.split_text(page.content)
            
            for chunk_text in page_chunks:
                # Skip empty chunks
                if not chunk_text.strip():
                    continue
                
                # Create unique chunk ID
                chunk_id = f"{document.filename}::page{page.page_number}::chunk{chunk_index}"
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    source_filename=document.filename,
                    source_page=page.page_number,
                    chunk_index=chunk_index,
                    metadata={
                        "file_type": document.file_type,
                        "total_pages": document.total_pages,
                        "original_char_count": document.total_chars
                    }
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        logger.info(
            "Document chunked successfully",
            filename=document.filename,
            total_chunks=len(chunks),
            avg_chunk_size=sum(c.char_count for c in chunks) / len(chunks) if chunks else 0
        )
        
        return chunks
    
    def chunk_text(
        self,
        text: str,
        source_filename: str = "unknown",
        source_page: int = 1
    ) -> List[DocumentChunk]:
        """
        Chunk raw text (convenience method).
        
        Useful for chunking text that wasn't loaded through DocumentParser.
        
        Args:
            text: Raw text to chunk
            source_filename: Filename for metadata
            source_page: Page number for metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks: List[DocumentChunk] = []
        text_chunks = self._splitter.split_text(text)
        
        for idx, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
                
            chunk_id = f"{source_filename}::page{source_page}::chunk{idx}"
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                char_count=len(chunk_text),
                source_filename=source_filename,
                source_page=source_page,
                chunk_index=idx
            ))
        
        return chunks
    
    def estimate_chunks(self, char_count: int) -> int:
        """
        Estimate number of chunks for a document.
        
        Useful for progress indicators and capacity planning.
        """
        if char_count <= self.chunk_size:
            return 1
        
        # Account for overlap
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        return max(1, (char_count - self.chunk_overlap) // effective_chunk_size + 1)
