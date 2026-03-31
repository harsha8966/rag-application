"""
Ask API Schemas - Request/Response Models for Question Answering

Pydantic models for the /ask endpoint.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """Information about a source chunk used in the answer."""
    source: str = Field(..., description="Source document filename")
    page: int = Field(..., description="Page number in source document")
    score: float = Field(..., description="Relevance score (0-1)")
    preview: str = Field(..., description="Text preview of the chunk")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "employee_handbook.pdf",
                "page": 12,
                "score": 0.892,
                "preview": "The company provides 15 days of paid time off annually..."
            }
        }


class ConfidenceScore(BaseModel):
    """Detailed confidence score breakdown."""
    overall: float = Field(..., ge=0, le=1, description="Overall confidence (0-1)")
    level: str = Field(..., description="Confidence level (high/medium/low)")
    percentage: int = Field(..., ge=0, le=100, description="Confidence as percentage")
    components: Dict[str, float] = Field(..., description="Component score breakdown")
    explanation: str = Field(..., description="Human-readable explanation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall": 0.847,
                "level": "high",
                "percentage": 85,
                "components": {
                    "retrieval": 0.892,
                    "agreement": 0.823,
                    "coverage": 0.786
                },
                "explanation": "High confidence answer based on strong document matches."
            }
        }


class AskRequest(BaseModel):
    """Request body for asking a question."""
    question: str = Field(
        ..., 
        min_length=3,
        max_length=1000,
        description="The question to ask"
    )
    use_mmr: bool = Field(
        default=True,
        description="Use MMR for diverse retrieval"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the company's remote work policy?",
                "use_mmr": True,
                "top_k": 5
            }
        }


class AskResponse(BaseModel):
    """Response containing the answer and metadata."""
    success: bool = Field(default=True)
    answer: str = Field(..., description="Generated answer")
    confidence: ConfidenceScore = Field(..., description="Confidence score details")
    sources: List[SourceChunk] = Field(..., description="Source chunks used")
    
    # Performance metrics
    retrieval_time_ms: float = Field(..., description="Time for retrieval")
    generation_time_ms: float = Field(..., description="Time for LLM generation")
    total_time_ms: float = Field(..., description="Total processing time")
    
    # Token usage
    tokens_used: Dict[str, int] = Field(..., description="Token usage breakdown")
    
    # For feedback correlation
    query_id: str = Field(..., description="Unique ID for this query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "answer": "According to the employee handbook, the company offers a flexible remote work policy. Employees may work remotely up to 3 days per week with manager approval. [Source: employee_handbook.pdf, Page 12]",
                "confidence": {
                    "overall": 0.847,
                    "level": "high",
                    "percentage": 85,
                    "components": {
                        "retrieval": 0.892,
                        "agreement": 0.823,
                        "coverage": 0.786
                    },
                    "explanation": "High confidence answer based on strong document matches."
                },
                "sources": [
                    {
                        "source": "employee_handbook.pdf",
                        "page": 12,
                        "score": 0.892,
                        "preview": "Remote Work Policy: Employees may work remotely..."
                    }
                ],
                "retrieval_time_ms": 45.2,
                "generation_time_ms": 1234.5,
                "total_time_ms": 1279.7,
                "tokens_used": {
                    "prompt": 1250,
                    "completion": 156,
                    "total": 1406
                },
                "query_id": "q_abc123def456"
            }
        }


class AskError(BaseModel):
    """Error response for failed questions."""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "RetrievalError",
                "message": "No documents have been uploaded yet"
            }
        }
