"""
Ask API Route - Question Answering Endpoint

The main RAG endpoint that:
1. Retrieves relevant document chunks
2. Constructs a strict prompt
3. Generates an answer using Gemini
4. Calculates confidence score
5. Returns answer with sources
"""

import re
import time
import uuid
from typing import List, Optional, Set, Tuple

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.ask import (
    AskRequest,
    AskResponse,
    AskError,
    SourceChunk,
    ConfidenceScore
)
from app.core.embeddings import FAISSVectorStore
from app.core.retrieval import DocumentRetriever
from app.core.reasoning import GeminiLLMClient, ConfidenceCalculator
from app.config import get_settings
from app.utils.exceptions import RetrievalError, LLMError
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ask", tags=["Question Answering"])

# Cache for query context (for feedback correlation)
# In production, use Redis or similar
_query_cache: dict = {}


def _extract_cited_sources(answer: str) -> Set[Tuple[str, int]]:
    """
    Parse the LLM answer for [Source: filename, Page X] citations
    and return a set of (filename, page_number) tuples that were
    actually referenced, so only genuinely cited sources are shown.
    """
    citation_pattern = r'\[Source:\s*(.+?),\s*Page\s*(\d+)\]'
    cited = set()
    for match in re.finditer(citation_pattern, answer):
        filename = match.group(1).strip()
        page = int(match.group(2))
        cited.add((filename, page))
    return cited


def _ensure_document_diversity(
    chunks: List,
    vector_store: "FAISSVectorStore",
    top_k: int
) -> List:
    """
    Guarantee that every uploaded document is represented in the
    retrieved chunks.  When a broad query (e.g. "summarize key points")
    causes similarity search to return results from only one document,
    this fills in at least one chunk per missing document so the LLM
    can reference all sources.

    Reads directly from the vector store metadata to completely bypass
    similarity thresholds that could silently discard chunks from
    less-similar documents.
    """
    all_documents = set(vector_store.get_stats().get("document_ids", []))
    if len(all_documents) <= 1:
        return chunks

    represented = {c.source_filename for c in chunks}
    missing_docs = all_documents - represented

    if not missing_docs:
        return chunks

    slots_per_missing = max(1, top_k // len(all_documents))

    from app.core.retrieval.retriever import RetrievedChunk

    for doc in missing_docs:
        # Pull directly from stored metadata — no search, no threshold.
        # Metadata is ordered by ingestion (page 1, 2, 3…), so taking
        # the first entries gives us early-page content which tends to
        # contain introductions and key summaries.
        doc_metadata = [
            m for m in vector_store._metadata
            if m.get("source_filename") == doc
        ][:slots_per_missing]

        for meta in doc_metadata:
            chunks.append(RetrievedChunk(
                content=meta.get("content", ""),
                score=0.5,
                source_filename=meta.get("source_filename", "Unknown"),
                source_page=meta.get("source_page", 1),
                chunk_id=meta.get("chunk_id", ""),
                metadata=meta
            ))

    logger.info(
        "Document diversity enforced",
        missing_docs=list(missing_docs),
        added_chunks=sum(
            1 for c in chunks if c.source_filename in missing_docs
        )
    )

    return chunks


@router.post(
    "",
    response_model=AskResponse,
    responses={
        400: {"model": AskError, "description": "Invalid request"},
        500: {"model": AskError, "description": "Processing error"}
    },
    summary="Ask a question",
    description="Ask a question and get an answer based on uploaded documents"
)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Process a question and return an answer from documents.
    
    Process:
    1. Validate question
    2. Retrieve relevant chunks (similarity or MMR)
    3. Generate answer with Gemini
    4. Calculate confidence score
    5. Return answer with sources and confidence
    """
    total_start = time.time()
    settings = get_settings()
    
    # Generate unique query ID for feedback tracking
    query_id = f"q_{uuid.uuid4().hex[:12]}"
    
    logger.info(
        "Processing question",
        query_id=query_id,
        question_length=len(request.question)
    )
    
    try:
        # Initialize vector store
        vector_store = FAISSVectorStore()
        
        # Check if we have any documents
        if vector_store.is_empty():
            raise RetrievalError(
                "No documents have been uploaded yet",
                details={"hint": "Upload documents using the /upload endpoint first"}
            )
        
        # Initialize retriever
        retriever = DocumentRetriever(vector_store)
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        
        top_k = request.top_k or settings.top_k_results
        
        if request.use_mmr:
            # MMR for diverse results
            chunks = retriever.retrieve_mmr(
                query=request.question,
                k=top_k,
                lambda_mult=settings.mmr_diversity_score
            )
        else:
            # Standard similarity search
            chunks = retriever.retrieve(
                query=request.question,
                k=top_k
            )
        
        # Ensure all uploaded documents are represented in the results
        chunks = _ensure_document_diversity(chunks, vector_store, top_k)

        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Handle case where no relevant chunks found
        if not chunks:
            logger.warning("No relevant chunks found", query_id=query_id)
            
            # Return a clear "no information" response
            confidence = ConfidenceScore(
                overall=0.0,
                level="low",
                percentage=0,
                components={"retrieval": 0.0, "agreement": 0.0, "coverage": 0.0},
                explanation="No relevant information found in the uploaded documents."
            )
            
            return AskResponse(
                success=True,
                answer="I don't have enough information in the uploaded documents to answer this question. Please try rephrasing your question or upload additional relevant documents.",
                confidence=confidence,
                sources=[],
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0,
                total_time_ms=(time.time() - total_start) * 1000,
                tokens_used={"prompt": 0, "completion": 0, "total": 0},
                query_id=query_id
            )
        
        # Prepare context for LLM
        context_chunks = [chunk.to_context_string() for chunk in chunks]
        
        # When context spans multiple documents, prepend an instruction so
        # the LLM addresses content from every document, not just one.
        unique_docs = {c.source_filename for c in chunks}
        if len(unique_docs) > 1:
            doc_list = ", ".join(sorted(unique_docs))
            context_chunks.insert(0, (
                f"[IMPORTANT: The following context contains information from "
                f"{len(unique_docs)} documents: {doc_list}. "
                f"You MUST include information from ALL of these documents "
                f"in your answer, citing each with [Source: filename, Page X].]"
            ))
        
        # Generate answer with Gemini
        generation_start = time.time()
        llm_client = GeminiLLMClient()
        
        llm_response = llm_client.generate_answer(
            question=request.question,
            context_chunks=context_chunks
        )
        
        generation_time = (time.time() - generation_start) * 1000
        
        # Calculate confidence score
        confidence_calc = ConfidenceCalculator()
        chunk_scores = [chunk.score for chunk in chunks]
        chunk_sources = [chunk.source_filename for chunk in chunks]
        
        confidence_result = confidence_calc.calculate(
            chunk_scores=chunk_scores,
            chunk_sources=chunk_sources,
            query=request.question,
            answer=llm_response.answer
        )
        
        # Filter sources to only those the LLM actually cited in the answer
        cited_sources = _extract_cited_sources(llm_response.answer)
        if cited_sources:
            cited_chunks = [
                chunk for chunk in chunks
                if (chunk.source_filename, chunk.source_page) in cited_sources
            ]
        else:
            # Fallback: if no citations were parsed (e.g. unexpected format),
            # return all retrieved chunks so the user still sees context
            cited_chunks = chunks

        sources = [
            SourceChunk(
                source=chunk.source_filename,
                page=chunk.source_page,
                score=round(chunk.score, 3),
                preview=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            )
            for chunk in cited_chunks
        ]
        
        # Format confidence for response
        confidence = ConfidenceScore(
            overall=confidence_result.overall,
            level=confidence_result.level,
            percentage=confidence_result.display_percentage,
            components=confidence_result.to_dict()["components"],
            explanation=confidence_result.explanation
        )
        
        total_time = (time.time() - total_start) * 1000
        
        # Cache query context for feedback
        _query_cache[query_id] = {
            "question": request.question,
            "answer": llm_response.answer,
            "confidence_score": confidence_result.overall,
            "chunks": [{"content": c.content, "source": c.source_filename, "page": c.source_page, "score": c.score} for c in chunks],
            "sources": [c.source_filename for c in chunks],
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": generation_time
        }
        
        logger.info(
            "Question answered successfully",
            query_id=query_id,
            confidence=confidence_result.overall,
            sources_count=len(sources),
            total_time_ms=total_time
        )
        
        return AskResponse(
            success=True,
            answer=llm_response.answer,
            confidence=confidence,
            sources=sources,
            retrieval_time_ms=round(retrieval_time, 2),
            generation_time_ms=round(generation_time, 2),
            total_time_ms=round(total_time, 2),
            tokens_used={
                "prompt": llm_response.prompt_tokens,
                "completion": llm_response.completion_tokens,
                "total": llm_response.total_tokens
            },
            query_id=query_id
        )
        
    except RetrievalError as e:
        logger.error("Retrieval error", error=str(e), query_id=query_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "RetrievalError",
                "message": str(e),
                "details": e.details
            }
        )
    except LLMError as e:
        logger.error("LLM error", error=str(e), query_id=query_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "LLMError",
                "message": str(e),
                "details": e.details
            }
        )
    except Exception as e:
        logger.error("Unexpected error", error=str(e), query_id=query_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalError",
                "message": "An unexpected error occurred",
                "details": {"error": str(e)}
            }
        )


def get_query_context(query_id: str) -> Optional[dict]:
    """
    Get cached query context for feedback.
    
    Called by feedback endpoint to correlate feedback with queries.
    """
    return _query_cache.get(query_id)
