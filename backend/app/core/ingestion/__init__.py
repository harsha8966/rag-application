"""
Document Ingestion Module

This module handles the complete document ingestion pipeline:
1. Parsing - Extract text from PDF/TXT files
2. Chunking - Split text into optimal-sized chunks
3. Metadata - Attach source information for citations

Why separate these concerns?
- Each step can be tested independently
- Easy to add new file format support
- Chunking strategy can be tuned without affecting parsing
"""

from app.core.ingestion.parser import DocumentParser
from app.core.ingestion.chunker import DocumentChunker
from app.core.ingestion.metadata import MetadataExtractor

__all__ = ["DocumentParser", "DocumentChunker", "MetadataExtractor"]
