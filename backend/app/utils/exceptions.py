"""
Custom Exceptions for Enterprise RAG Assistant

Why custom exceptions?
- Clear error categorization for debugging
- Enables specific error handling at API layer
- Better error messages for users
- Facilitates monitoring and alerting
"""


class RAGException(Exception):
    """
    Base exception for all RAG-related errors.
    
    All custom exceptions inherit from this, allowing:
    - Catch-all handling: except RAGException
    - Specific handling: except DocumentProcessingError
    """
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class DocumentProcessingError(RAGException):
    """
    Raised when document parsing or chunking fails.
    
    Examples:
    - Corrupted PDF file
    - Unsupported file format
    - Empty document
    - Encoding issues in text files
    """
    pass


class EmbeddingError(RAGException):
    """
    Raised when embedding generation fails.
    
    Examples:
    - Gemini API rate limit exceeded
    - Invalid API key
    - Network timeout
    - Text too long for embedding model
    """
    pass


class RetrievalError(RAGException):
    """
    Raised when vector search fails.
    
    Examples:
    - FAISS index not found
    - Corrupted index file
    - No documents indexed yet
    """
    pass


class LLMError(RAGException):
    """
    Raised when LLM generation fails.
    
    Examples:
    - Gemini API error
    - Context too long
    - Content filtered by safety settings
    """
    pass


class ValidationError(RAGException):
    """
    Raised when input validation fails.
    
    Examples:
    - File too large
    - Unsupported file type
    - Empty query
    - Invalid feedback value
    """
    pass
