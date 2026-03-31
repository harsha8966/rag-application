"""
Feedback API Schemas - Request/Response Models for User Feedback

Pydantic models for the /feedback endpoint.
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Types of feedback users can provide."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"


class FeedbackRequest(BaseModel):
    """Request body for submitting feedback."""
    query_id: str = Field(
        ...,
        description="Query ID from the original /ask response"
    )
    feedback_type: FeedbackType = Field(
        ...,
        description="Type of feedback"
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional free-text comment"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_id": "q_abc123def456",
                "feedback_type": "positive",
                "comment": "The answer was accurate and well-sourced"
            }
        }


class FeedbackResponse(BaseModel):
    """Response confirming feedback submission."""
    success: bool = Field(default=True)
    message: str = Field(..., description="Confirmation message")
    feedback_id: str = Field(..., description="Unique feedback ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Thank you for your feedback!",
                "feedback_id": "fb_xyz789"
            }
        }


class FeedbackStatsResponse(BaseModel):
    """Response with feedback statistics."""
    total_feedback: int = Field(..., description="Total feedback entries")
    positive_rate: float = Field(..., description="Percentage of positive feedback")
    negative_rate: float = Field(..., description="Percentage of negative feedback")
    average_confidence: float = Field(..., description="Average confidence of rated answers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_feedback": 150,
                "positive_rate": 0.78,
                "negative_rate": 0.15,
                "average_confidence": 0.72
            }
        }
