"""
Feedback API Route - User Feedback Collection Endpoint

Handles collection of user feedback on RAG responses
for quality monitoring and improvement.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.feedback import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    FeedbackType as SchemaFeedbackType
)
from app.core.feedback import FeedbackStore, FeedbackEntry
from app.core.feedback.feedback_store import FeedbackType
from app.api.routes.ask import get_query_context
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    summary="Submit feedback",
    description="Submit feedback on a previous answer"
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Record user feedback for a previous query.
    
    Feedback is used for:
    - Quality monitoring
    - Identifying problem areas
    - Improving retrieval and prompts
    """
    logger.info(
        "Feedback received",
        query_id=request.query_id,
        feedback_type=request.feedback_type.value
    )
    
    # Get original query context
    query_context = get_query_context(request.query_id)
    
    if not query_context:
        # Still record feedback, but with limited context
        logger.warning(
            "Query context not found",
            query_id=request.query_id
        )
        query_context = {
            "question": "Unknown (query expired from cache)",
            "answer": "Unknown",
            "confidence_score": 0.0,
            "chunks": [],
            "sources": [],
            "retrieval_time_ms": None,
            "generation_time_ms": None
        }
    
    # Map schema feedback type to internal type
    feedback_type_map = {
        SchemaFeedbackType.POSITIVE: FeedbackType.POSITIVE,
        SchemaFeedbackType.NEGATIVE: FeedbackType.NEGATIVE,
        SchemaFeedbackType.PARTIAL: FeedbackType.PARTIAL,
        SchemaFeedbackType.IRRELEVANT: FeedbackType.IRRELEVANT
    }
    internal_feedback_type = feedback_type_map[request.feedback_type]
    
    # Store feedback
    feedback_store = FeedbackStore()
    entry = feedback_store.record_feedback(
        feedback_type=internal_feedback_type,
        question=query_context["question"],
        answer=query_context["answer"],
        confidence_score=query_context["confidence_score"],
        retrieved_chunks=query_context["chunks"],
        sources_used=query_context["sources"],
        comment=request.comment,
        retrieval_latency_ms=query_context.get("retrieval_time_ms"),
        llm_latency_ms=query_context.get("generation_time_ms")
    )
    
    # Determine response message based on feedback type
    messages = {
        FeedbackType.POSITIVE: "Thank you for the positive feedback!",
        FeedbackType.NEGATIVE: "Thank you for your feedback. We'll use it to improve.",
        FeedbackType.PARTIAL: "Thank you for letting us know. We'll work on improving accuracy.",
        FeedbackType.IRRELEVANT: "Thank you for the feedback. We'll review the retrieval quality."
    }
    
    return FeedbackResponse(
        success=True,
        message=messages.get(internal_feedback_type, "Thank you for your feedback!"),
        feedback_id=entry.feedback_id
    )


@router.get(
    "/stats",
    response_model=FeedbackStatsResponse,
    summary="Get feedback statistics",
    description="Get aggregate statistics on user feedback"
)
async def get_feedback_stats() -> FeedbackStatsResponse:
    """
    Get feedback statistics for monitoring.
    
    Useful for:
    - Quality dashboards
    - Identifying trends
    - Measuring improvement over time
    """
    try:
        feedback_store = FeedbackStore()
        stats = feedback_store.get_statistics()
        
        return FeedbackStatsResponse(
            total_feedback=stats["total_feedback"],
            positive_rate=stats["positive_rate"],
            negative_rate=stats["negative_rate"],
            average_confidence=stats["average_confidence"]
        )
    except Exception as e:
        logger.error("Failed to get feedback stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": str(e)}
        )
