"""
Enterprise RAG Assistant - FastAPI Application Entry Point

This is the main application file that:
- Configures the FastAPI application
- Sets up CORS for frontend communication
- Registers all API routes
- Configures logging and error handling
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import upload_router, ask_router, feedback_router
from app.config import get_settings
from app.utils.logging import setup_logging, get_logger
from app.utils.exceptions import RAGException

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    
    Startup:
    - Log application start
    - Validate configuration
    
    Shutdown:
    - Clean up resources
    """
    # Startup
    settings = get_settings()
    logger.info(
        "Starting Enterprise RAG Assistant",
        env=settings.app_env,
        debug=settings.debug
    )
    
    # Validate API key is configured
    if not settings.google_api_key:
        logger.warning(
            "Google API key not configured - LLM features will not work"
        )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enterprise RAG Assistant")


# Create FastAPI application
app = FastAPI(
    title="Enterprise RAG Assistant API",
    description="""
    A production-grade Retrieval Augmented Generation (RAG) API that answers 
    questions ONLY from uploaded internal documents while minimizing hallucinations.
    
    ## Features
    
    - **Document Upload**: Upload PDF and TXT documents for indexing
    - **Question Answering**: Ask questions and get grounded answers
    - **Confidence Scores**: Every answer includes a confidence indicator
    - **Source Citations**: Answers include source document references
    - **Feedback Collection**: Track user satisfaction for improvement
    
    ## Architecture
    
    - **Embeddings**: Google Gemini text-embedding-004
    - **Vector Store**: FAISS for fast similarity search
    - **LLM**: Google Gemini 1.5 Pro for answer generation
    - **Retrieval**: MMR-based retrieval for diverse results
    
    ## Hallucination Prevention
    
    - Strict prompting forces answers ONLY from provided context
    - LLM explicitly instructed to say "I don't know" when info is missing
    - Confidence scores help users assess answer reliability
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler for RAG exceptions
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle all RAG-specific exceptions."""
    logger.error(
        "RAG exception",
        error_type=exc.__class__.__name__,
        message=exc.message,
        details=exc.details
    )
    return JSONResponse(
        status_code=500,
        content=exc.to_dict()
    )


# Register routers
app.include_router(upload_router)
app.include_router(ask_router)
app.include_router(feedback_router)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns basic application status.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "enterprise-rag-assistant"
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Enterprise RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
