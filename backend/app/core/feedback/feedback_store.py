"""
Feedback Storage - User Feedback Collection for Quality Improvement

This module handles the collection and storage of user feedback on RAG responses.

=== HOW FEEDBACK IMPROVES SYSTEM QUALITY WITHOUT RETRAINING ===

1. RETRIEVAL TUNING
   - Negative feedback on answers reveals poor retrievals
   - Identify queries that need better chunk matching
   - Adjust similarity thresholds based on feedback patterns

2. PROMPT REFINEMENT
   - Feedback shows where prompts fail to constrain LLM
   - Identify cases where LLM ignores context
   - Improve prompt templates iteratively

3. DOCUMENT GAPS
   - Repeated negative feedback on topics = missing documents
   - Guides content team on what to upload
   - Identifies outdated information

4. QUALITY MONITORING
   - Track satisfaction over time
   - Alert on quality degradation
   - A/B test different configurations

5. ANALYTICS
   - Most asked questions
   - Hardest questions (lowest confidence)
   - User patterns and needs

=== STORAGE DESIGN ===

Using JSON files for simplicity and portability.
Production systems might use:
- PostgreSQL for structured queries
- Elasticsearch for full-text search
- Data warehouse for analytics

The interface is designed for easy migration to other backends.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback users can provide."""
    POSITIVE = "positive"      # Thumbs up - answer was helpful
    NEGATIVE = "negative"      # Thumbs down - answer was wrong/unhelpful
    PARTIAL = "partial"        # Partially correct
    IRRELEVANT = "irrelevant"  # Answer didn't address the question


@dataclass
class FeedbackEntry:
    """
    Complete feedback record with full context.
    
    Stores everything needed to analyze why an answer
    succeeded or failed.
    """
    # Identification
    feedback_id: str
    timestamp: str
    
    # User feedback
    feedback_type: FeedbackType
    comment: Optional[str] = None  # Optional free-text feedback
    
    # Original interaction context
    question: str = ""
    answer: str = ""
    confidence_score: float = 0.0
    
    # Retrieved chunks (for debugging retrieval)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Source information
    sources_used: List[str] = field(default_factory=list)
    
    # Session tracking (optional)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Performance metrics
    retrieval_latency_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["feedback_type"] = self.feedback_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEntry":
        """Create from dictionary."""
        data["feedback_type"] = FeedbackType(data["feedback_type"])
        return cls(**data)


class FeedbackStore:
    """
    Persistent storage for user feedback.
    
    Features:
    - Append-only logging (preserves all feedback)
    - Efficient bulk retrieval
    - Filtering by type, date, confidence
    - Analytics helpers
    
    Usage:
        store = FeedbackStore()
        
        # Record feedback
        entry = store.record_feedback(
            feedback_type=FeedbackType.POSITIVE,
            question="What is the return policy?",
            answer="The return policy allows...",
            confidence_score=0.85,
            retrieved_chunks=[...],
            sources_used=["policy.pdf"]
        )
        
        # Analyze feedback
        negative = store.get_negative_feedback(limit=100)
        stats = store.get_statistics()
    """
    
    FEEDBACK_FILENAME = "feedback_log.jsonl"  # JSON Lines format
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize feedback store.
        
        Args:
            storage_path: Directory for feedback storage.
                         If None, uses config default.
        """
        settings = get_settings()
        self.storage_path = storage_path or settings.feedback_directory
        self.storage_path = Path(self.storage_path)
        
        # Ensure directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.storage_path / self.FEEDBACK_FILENAME
        
        logger.info(
            "FeedbackStore initialized",
            path=str(self.feedback_file)
        )
    
    def record_feedback(
        self,
        feedback_type: FeedbackType,
        question: str,
        answer: str,
        confidence_score: float,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        sources_used: Optional[List[str]] = None,
        comment: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        retrieval_latency_ms: Optional[float] = None,
        llm_latency_ms: Optional[float] = None
    ) -> FeedbackEntry:
        """
        Record a new feedback entry.
        
        Args:
            feedback_type: Type of feedback (positive/negative/etc)
            question: Original user question
            answer: Generated answer
            confidence_score: System's confidence score
            retrieved_chunks: Chunks used for context
            sources_used: Source document names
            comment: Optional user comment
            session_id: Optional session tracking
            user_id: Optional user tracking
            retrieval_latency_ms: Retrieval time
            llm_latency_ms: LLM generation time
            
        Returns:
            Created FeedbackEntry
        """
        entry = FeedbackEntry(
            feedback_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            feedback_type=feedback_type,
            question=question,
            answer=answer,
            confidence_score=confidence_score,
            retrieved_chunks=retrieved_chunks or [],
            sources_used=sources_used or [],
            comment=comment,
            session_id=session_id,
            user_id=user_id,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms
        )
        
        # Append to file (JSON Lines format)
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        
        logger.info(
            "Feedback recorded",
            feedback_id=entry.feedback_id,
            type=feedback_type.value,
            confidence=confidence_score
        )
        
        return entry
    
    def get_all_feedback(self) -> List[FeedbackEntry]:
        """Retrieve all feedback entries."""
        if not self.feedback_file.exists():
            return []
        
        entries = []
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(FeedbackEntry.from_dict(data))
        
        return entries
    
    def get_feedback_by_type(
        self,
        feedback_type: FeedbackType,
        limit: Optional[int] = None
    ) -> List[FeedbackEntry]:
        """Get feedback filtered by type."""
        all_feedback = self.get_all_feedback()
        filtered = [f for f in all_feedback if f.feedback_type == feedback_type]
        
        # Sort by timestamp descending (most recent first)
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_negative_feedback(self, limit: int = 100) -> List[FeedbackEntry]:
        """Get negative feedback for analysis."""
        return self.get_feedback_by_type(FeedbackType.NEGATIVE, limit)
    
    def get_low_confidence_feedback(
        self,
        threshold: float = 0.5,
        limit: int = 100
    ) -> List[FeedbackEntry]:
        """Get feedback for low-confidence answers."""
        all_feedback = self.get_all_feedback()
        low_conf = [f for f in all_feedback if f.confidence_score < threshold]
        low_conf.sort(key=lambda x: x.confidence_score)
        return low_conf[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate feedback statistics.
        
        Returns summary metrics useful for dashboards and monitoring.
        """
        all_feedback = self.get_all_feedback()
        
        if not all_feedback:
            return {
                "total_feedback": 0,
                "positive_rate": 0.0,
                "negative_rate": 0.0,
                "average_confidence": 0.0,
                "feedback_by_type": {}
            }
        
        total = len(all_feedback)
        
        # Count by type
        by_type = {}
        for entry in all_feedback:
            type_name = entry.feedback_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        # Calculate rates
        positive = by_type.get("positive", 0)
        negative = by_type.get("negative", 0)
        
        positive_rate = positive / total if total > 0 else 0
        negative_rate = negative / total if total > 0 else 0
        
        # Average confidence
        avg_confidence = sum(f.confidence_score for f in all_feedback) / total
        
        # Confidence correlation with feedback
        positive_entries = [f for f in all_feedback if f.feedback_type == FeedbackType.POSITIVE]
        negative_entries = [f for f in all_feedback if f.feedback_type == FeedbackType.NEGATIVE]
        
        avg_positive_conf = sum(f.confidence_score for f in positive_entries) / len(positive_entries) if positive_entries else 0
        avg_negative_conf = sum(f.confidence_score for f in negative_entries) / len(negative_entries) if negative_entries else 0
        
        return {
            "total_feedback": total,
            "positive_rate": round(positive_rate, 3),
            "negative_rate": round(negative_rate, 3),
            "average_confidence": round(avg_confidence, 3),
            "feedback_by_type": by_type,
            "avg_confidence_positive": round(avg_positive_conf, 3),
            "avg_confidence_negative": round(avg_negative_conf, 3)
        }
    
    def get_problem_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Identify queries that frequently receive negative feedback.
        
        Useful for:
        - Identifying document gaps
        - Finding retrieval issues
        - Prioritizing improvements
        """
        negative = self.get_negative_feedback(limit=500)
        
        # Group by question (simplified - real impl would use similarity)
        question_feedback: Dict[str, List[FeedbackEntry]] = {}
        for entry in negative:
            # Normalize question for grouping
            normalized = entry.question.lower().strip()
            if normalized not in question_feedback:
                question_feedback[normalized] = []
            question_feedback[normalized].append(entry)
        
        # Sort by frequency
        problems = [
            {
                "question": entries[0].question,
                "count": len(entries),
                "avg_confidence": sum(e.confidence_score for e in entries) / len(entries),
                "sources": list(set(s for e in entries for s in e.sources_used))
            }
            for entries in question_feedback.values()
        ]
        
        problems.sort(key=lambda x: x["count"], reverse=True)
        
        return problems[:limit]
