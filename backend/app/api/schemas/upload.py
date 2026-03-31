"""
Upload API Schemas - Request/Response Models for Document Upload

Pydantic models for type-safe API contracts.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Information about a successfully processed document."""
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (pdf, txt)")
    total_pages: int = Field(..., description="Number of pages in document")
    total_chars: int = Field(..., description="Total character count")
    total_chunks: int = Field(..., description="Number of chunks created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "company_policy_pdf_a1b2c3d4",
                "filename": "company_policy.pdf",
                "file_type": "pdf",
                "total_pages": 15,
                "total_chars": 45000,
                "total_chunks": 62
            }
        }


class UploadResponse(BaseModel):
    """Response for successful document upload."""
    success: bool = Field(default=True)
    message: str = Field(..., description="Status message")
    document: DocumentInfo = Field(..., description="Processed document info")
    processing_time_ms: float = Field(..., description="Time to process document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document uploaded and indexed successfully",
                "document": {
                    "document_id": "company_policy_pdf_a1b2c3d4",
                    "filename": "company_policy.pdf",
                    "file_type": "pdf",
                    "total_pages": 15,
                    "total_chars": 45000,
                    "total_chunks": 62
                },
                "processing_time_ms": 2345.67
            }
        }


class UploadError(BaseModel):
    """Error response for failed upload."""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error description")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "File type not supported",
                "details": {
                    "file_extension": ".docx",
                    "supported_extensions": [".pdf", ".txt"]
                }
            }
        }


class DocumentListResponse(BaseModel):
    """Response listing all indexed documents."""
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total chunks across all documents")
    documents: List[str] = Field(..., description="List of document filenames")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 5,
                "total_chunks": 234,
                "documents": ["policy.pdf", "handbook.pdf", "faq.txt"]
            }
        }


class DeleteDocumentResponse(BaseModel):
    """Response for document deletion."""
    success: bool = Field(default=True)
    message: str = Field(..., description="Status message")
    document_id: str = Field(..., description="Deleted document ID")
    chunks_removed: int = Field(..., description="Number of chunks removed")
