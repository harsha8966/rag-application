"""
Upload API Route - Document Upload and Indexing Endpoint

Handles:
- File upload validation
- Document parsing
- Chunking
- Embedding generation
- Vector store indexing
"""

import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.api.schemas.upload import (
    UploadResponse,
    UploadError,
    DocumentInfo,
    DocumentListResponse,
    DeleteDocumentResponse
)
from app.core.ingestion import DocumentParser, DocumentChunker, MetadataExtractor
from app.core.embeddings import FAISSVectorStore
from app.config import get_settings
from app.utils.exceptions import DocumentProcessingError, ValidationError
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/upload", tags=["Document Upload"])


@router.post(
    "",
    response_model=UploadResponse,
    responses={
        400: {"model": UploadError, "description": "Invalid file"},
        500: {"model": UploadError, "description": "Processing error"}
    },
    summary="Upload a document",
    description="Upload a PDF or TXT document to be indexed for RAG"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF or TXT)")
) -> UploadResponse:
    """
    Upload and index a document for RAG.
    
    Process:
    1. Validate file type and size
    2. Save to temp location
    3. Parse document
    4. Chunk text
    5. Generate embeddings
    6. Store in vector database
    """
    start_time = time.time()
    settings = get_settings()
    
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": f"File type '{file_extension}' not supported",
                "details": {
                    "file_extension": file_extension,
                    "supported_extensions": settings.allowed_extensions_list
                }
            }
        )
    
    # Validate file size
    file_content = await file.read()
    if len(file_content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": f"File too large. Maximum size is {settings.max_file_size_mb}MB",
                "details": {
                    "file_size_mb": len(file_content) / (1024 * 1024),
                    "max_size_mb": settings.max_file_size_mb
                }
            }
        )
    
    # Save file temporarily
    upload_dir = settings.upload_directory
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / file.filename
    
    try:
        # Write file
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        logger.info("File saved", filename=file.filename, size=len(file_content))
        
        # Parse document
        parser = DocumentParser()
        parsed_doc = parser.parse(temp_path)
        
        # Chunk document
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(parsed_doc)
        
        if not chunks:
            raise DocumentProcessingError(
                "No content extracted from document",
                details={"filename": file.filename}
            )
        
        # Extract metadata
        metadata_extractor = MetadataExtractor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        doc_metadata = metadata_extractor.extract_document_metadata(parsed_doc, chunks)
        
        # Enrich chunk metadata
        enriched_metadata = [
            metadata_extractor.enrich_chunk_metadata(chunk, doc_metadata)
            for chunk in chunks
        ]
        
        # Add content to metadata for retrieval
        for i, chunk in enumerate(chunks):
            enriched_metadata[i]["content"] = chunk.content
        
        # Store in vector database
        vector_store = FAISSVectorStore()
        vector_store.add_chunks(chunks, enriched_metadata)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            "Document indexed successfully",
            filename=file.filename,
            chunks=len(chunks),
            time_ms=processing_time
        )
        
        return UploadResponse(
            success=True,
            message="Document uploaded and indexed successfully",
            document=DocumentInfo(
                document_id=doc_metadata.document_id,
                filename=file.filename,
                file_type=doc_metadata.file_type,
                total_pages=doc_metadata.total_pages,
                total_chars=doc_metadata.total_chars,
                total_chunks=len(chunks)
            ),
            processing_time_ms=round(processing_time, 2)
        )
        
    except DocumentProcessingError as e:
        logger.error("Document processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "DocumentProcessingError",
                "message": str(e),
                "details": e.details
            }
        )
    except Exception as e:
        logger.error("Unexpected error during upload", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred",
                "details": {"error": str(e)}
            }
        )
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List indexed documents",
    description="Get a list of all documents in the vector store"
)
async def list_documents() -> DocumentListResponse:
    """List all indexed documents."""
    try:
        vector_store = FAISSVectorStore()
        stats = vector_store.get_stats()
        
        return DocumentListResponse(
            total_documents=stats["total_documents"],
            total_chunks=stats["total_vectors"],
            documents=stats["document_ids"]
        )
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": str(e)}
        )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document",
    description="Remove a document and its chunks from the vector store"
)
async def delete_document(document_id: str) -> DeleteDocumentResponse:
    """Delete a document from the index."""
    try:
        vector_store = FAISSVectorStore()
        chunks_removed = vector_store.delete_document(document_id)
        
        if chunks_removed == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "NotFound", "message": f"Document '{document_id}' not found"}
            )
        
        return DeleteDocumentResponse(
            success=True,
            message=f"Document deleted successfully",
            document_id=document_id,
            chunks_removed=chunks_removed
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": str(e)}
        )
