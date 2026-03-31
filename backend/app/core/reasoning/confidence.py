"""
Confidence Score Calculation for RAG Responses

This module calculates confidence scores to help users understand
how reliable an answer is likely to be.

=== HOW CONFIDENCE IMPROVES USER TRUST ===

1. TRANSPARENCY
   - Users know when to trust answers
   - High confidence = rely on it
   - Low confidence = verify manually

2. RISK MANAGEMENT
   - Enterprise users need to assess risk
   - Critical decisions require high confidence
   - Low confidence triggers human review

3. SYSTEM FEEDBACK
   - Identifies weak areas in document coverage
   - Guides document upload priorities
   - Helps improve retrieval tuning

=== WHY THIS MATTERS IN ENTERPRISE SETTINGS ===

1. ACCOUNTABILITY
   - Decisions based on RAG need justification
   - Confidence scores provide evidence
   - Audit trails for compliance

2. APPROPRIATE RELIANCE
   - Prevents over-trusting AI
   - Encourages verification when needed
   - Balances efficiency and accuracy

3. CONTINUOUS IMPROVEMENT
   - Track confidence over time
   - Identify topics needing better coverage
   - Measure system effectiveness

=== CONFIDENCE CALCULATION FACTORS ===

1. RETRIEVAL QUALITY
   - Top chunk similarity scores
   - Average score across chunks
   - Spread between best and worst

2. CHUNK AGREEMENT
   - Do multiple chunks support the answer?
   - Are they from different documents?
   - Consensus increases confidence

3. COVERAGE
   - How much of the question is addressed?
   - Are there information gaps?
   - Partial answers get partial confidence
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConfidenceScore:
    """
    Detailed confidence score with component breakdown.
    
    Provides:
    - Overall score (0-1)
    - Component scores for debugging
    - Human-readable explanation
    """
    overall: float  # 0-1, primary score to show users
    
    # Component scores (0-1 each)
    retrieval_score: float  # Based on similarity scores
    agreement_score: float  # Based on chunk agreement
    coverage_score: float   # Based on context coverage
    
    # Metadata
    explanation: str
    factors: Dict[str, Any]
    
    @property
    def level(self) -> str:
        """Get human-readable confidence level."""
        if self.overall >= 0.8:
            return "high"
        elif self.overall >= 0.5:
            return "medium"
        else:
            return "low"
    
    @property
    def display_percentage(self) -> int:
        """Get confidence as percentage for UI."""
        return round(self.overall * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "overall": round(self.overall, 3),
            "level": self.level,
            "percentage": self.display_percentage,
            "components": {
                "retrieval": round(self.retrieval_score, 3),
                "agreement": round(self.agreement_score, 3),
                "coverage": round(self.coverage_score, 3)
            },
            "explanation": self.explanation,
            "factors": self.factors
        }


class ConfidenceCalculator:
    """
    Calculates confidence scores for RAG responses.
    
    Uses multiple signals:
    1. Retrieval similarity scores
    2. Agreement between chunks
    3. Context coverage estimation
    
    Usage:
        calculator = ConfidenceCalculator()
        score = calculator.calculate(
            chunk_scores=[0.85, 0.78, 0.72],
            chunk_sources=["doc1.pdf", "doc1.pdf", "doc2.pdf"],
            query="What is the policy?"
        )
        print(f"Confidence: {score.overall}")
    """
    
    # Weights for component scores
    RETRIEVAL_WEIGHT = 0.5
    AGREEMENT_WEIGHT = 0.3
    COVERAGE_WEIGHT = 0.2
    
    # Thresholds
    HIGH_SCORE_THRESHOLD = 0.8
    LOW_SCORE_THRESHOLD = 0.5
    
    def __init__(
        self,
        retrieval_weight: float = 0.5,
        agreement_weight: float = 0.3,
        coverage_weight: float = 0.2
    ):
        """
        Initialize calculator with custom weights.
        
        Args:
            retrieval_weight: Weight for retrieval quality (0-1)
            agreement_weight: Weight for chunk agreement (0-1)
            coverage_weight: Weight for coverage estimation (0-1)
        """
        # Normalize weights
        total = retrieval_weight + agreement_weight + coverage_weight
        self.retrieval_weight = retrieval_weight / total
        self.agreement_weight = agreement_weight / total
        self.coverage_weight = coverage_weight / total
    
    def calculate(
        self,
        chunk_scores: List[float],
        chunk_sources: List[str],
        query: str,
        answer: Optional[str] = None
    ) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score.
        
        Args:
            chunk_scores: Similarity scores from retrieval
            chunk_sources: Source filenames for each chunk
            query: Original user query
            answer: Optional generated answer for coverage analysis
            
        Returns:
            ConfidenceScore with detailed breakdown
        """
        factors = {
            "num_chunks": len(chunk_scores),
            "num_sources": len(set(chunk_sources)),
            "query_length": len(query.split())
        }
        
        # Calculate component scores
        retrieval_score = self._calculate_retrieval_score(chunk_scores)
        agreement_score = self._calculate_agreement_score(chunk_scores, chunk_sources)
        coverage_score = self._calculate_coverage_score(chunk_scores, query, answer)
        
        # Weighted combination
        overall = (
            retrieval_score * self.retrieval_weight +
            agreement_score * self.agreement_weight +
            coverage_score * self.coverage_weight
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            overall, retrieval_score, agreement_score, coverage_score, factors
        )
        
        factors.update({
            "top_score": max(chunk_scores) if chunk_scores else 0,
            "avg_score": sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0,
            "score_spread": max(chunk_scores) - min(chunk_scores) if len(chunk_scores) > 1 else 0
        })
        
        score = ConfidenceScore(
            overall=overall,
            retrieval_score=retrieval_score,
            agreement_score=agreement_score,
            coverage_score=coverage_score,
            explanation=explanation,
            factors=factors
        )
        
        logger.debug(
            "Confidence calculated",
            overall=score.overall,
            level=score.level
        )
        
        return score
    
    def _calculate_retrieval_score(self, chunk_scores: List[float]) -> float:
        """
        Calculate retrieval quality score.
        
        Factors:
        - Top chunk score (most important)
        - Average score
        - Score decay (how much worse are lower chunks)
        """
        if not chunk_scores:
            return 0.0
        
        top_score = max(chunk_scores)
        avg_score = sum(chunk_scores) / len(chunk_scores)
        
        # Weighted: top score matters most
        # Formula: 60% top + 40% average
        retrieval = 0.6 * top_score + 0.4 * avg_score
        
        return min(1.0, retrieval)
    
    def _calculate_agreement_score(
        self,
        chunk_scores: List[float],
        chunk_sources: List[str]
    ) -> float:
        """
        Calculate agreement score based on chunk consistency.
        
        Higher score when:
        - Multiple high-scoring chunks
        - Chunks from different sources agree
        - Low variance in scores
        """
        if len(chunk_scores) < 2:
            return 0.5  # Neutral for single chunk
        
        # Count high-quality chunks
        high_quality = sum(1 for s in chunk_scores if s >= self.LOW_SCORE_THRESHOLD)
        quality_ratio = high_quality / len(chunk_scores)
        
        # Multi-source bonus (more sources = more agreement)
        unique_sources = len(set(chunk_sources))
        source_bonus = min(1.0, unique_sources / 3)  # Max bonus at 3+ sources
        
        # Score consistency (lower variance = higher agreement)
        if len(chunk_scores) > 1:
            variance = sum((s - sum(chunk_scores)/len(chunk_scores))**2 for s in chunk_scores) / len(chunk_scores)
            consistency = 1.0 - min(1.0, variance * 4)  # Scale variance to 0-1
        else:
            consistency = 0.5
        
        # Combine factors
        agreement = 0.4 * quality_ratio + 0.3 * source_bonus + 0.3 * consistency
        
        return min(1.0, agreement)
    
    def _calculate_coverage_score(
        self,
        chunk_scores: List[float],
        query: str,
        answer: Optional[str]
    ) -> float:
        """
        Estimate how well the context covers the query.
        
        Heuristics:
        - More chunks = potentially better coverage
        - Longer answer = more complete
        - High scores across chunks = good coverage
        """
        if not chunk_scores:
            return 0.0
        
        # Chunk count factor (diminishing returns after 3)
        chunk_factor = min(1.0, len(chunk_scores) / 3)
        
        # Score distribution (all high scores = good coverage)
        min_score = min(chunk_scores)
        score_floor = min_score / max(chunk_scores) if max(chunk_scores) > 0 else 0
        
        # Answer length factor (if provided)
        if answer:
            # Rough heuristic: longer answers often indicate more coverage
            # But cap it to avoid gaming
            word_count = len(answer.split())
            answer_factor = min(1.0, word_count / 100)
        else:
            answer_factor = 0.5  # Neutral if no answer
        
        coverage = 0.3 * chunk_factor + 0.4 * score_floor + 0.3 * answer_factor
        
        return min(1.0, coverage)
    
    def _generate_explanation(
        self,
        overall: float,
        retrieval: float,
        agreement: float,
        coverage: float,
        factors: Dict[str, Any]
    ) -> str:
        """Generate human-readable confidence explanation."""
        explanations = []
        
        # Overall assessment
        if overall >= 0.8:
            explanations.append("High confidence answer based on strong document matches.")
        elif overall >= 0.5:
            explanations.append("Moderate confidence - answer is likely correct but verify important details.")
        else:
            explanations.append("Low confidence - the documents may not contain complete information for this question.")
        
        # Component explanations
        if retrieval < 0.6:
            explanations.append("The retrieved documents have relatively low relevance scores.")
        
        if agreement < 0.5 and factors.get("num_chunks", 0) > 1:
            explanations.append("Retrieved chunks show varying levels of relevance.")
        
        if factors.get("num_sources", 0) > 1 and agreement >= 0.6:
            explanations.append("Multiple documents support this answer.")
        
        return " ".join(explanations)
    
    def calculate_simple(self, chunk_scores: List[float]) -> float:
        """
        Quick confidence calculation using only scores.
        
        Use when you just need a number, not full breakdown.
        """
        if not chunk_scores:
            return 0.0
        
        # Simple formula: 70% top score + 30% average
        top = max(chunk_scores)
        avg = sum(chunk_scores) / len(chunk_scores)
        
        return 0.7 * top + 0.3 * avg
