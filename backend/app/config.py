"""
Configuration Management for Enterprise RAG Assistant

This module centralizes all configuration using Pydantic Settings.
Why Pydantic Settings?
- Type-safe configuration with validation
- Automatic environment variable loading
- Clear documentation of all settings
- Easy testing with override capabilities
"""

from pathlib import Path
from typing import List
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings have sensible defaults for development.
    Production deployments should override via .env file or environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra env vars
    )
    
    # === API Keys ===
    google_api_key: str = Field(
        default="",
        description="Google Gemini API key for embeddings and LLM"
    )
    
    # === Application Settings ===
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    
    # === Server Configuration ===
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # === Vector Store Settings ===
    faiss_index_path: str = Field(default="./data/faiss_index")
    chunk_size: int = Field(
        default=750,
        description="Target chunk size in characters (roughly 700-800 tokens)"
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap between chunks to maintain context continuity"
    )
    
    # === Retrieval Settings ===
    top_k_results: int = Field(
        default=5,
        description="Number of chunks to retrieve for each query"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score to include a chunk (0-1)"
    )
    mmr_diversity_score: float = Field(
        default=0.3,
        description="MMR lambda - higher values favor diversity over similarity"
    )
    
    # === LLM Settings ===
    gemini_model: str = Field(
        default="gemini-1.5-pro",
        description="Gemini model for answer generation"
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Gemini model for text embeddings"
    )
    max_output_tokens: int = Field(default=2048)
    temperature: float = Field(
        default=0.1,
        description="Low temperature for factual, consistent responses"
    )
    
    # === File Upload Settings ===
    max_file_size_mb: int = Field(default=50)
    allowed_extensions: str = Field(default=".pdf,.txt")
    upload_dir: str = Field(default="./data/uploads")
    
    # === Feedback Storage ===
    feedback_dir: str = Field(default="./data/feedback")
    
    # === Computed Properties ===
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse comma-separated origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Parse comma-separated extensions into a list."""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes for file validation."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def faiss_index_directory(self) -> Path:
        """Get FAISS index path as Path object."""
        return Path(self.faiss_index_path)
    
    @property
    def upload_directory(self) -> Path:
        """Get upload directory as Path object."""
        return Path(self.upload_dir)
    
    @property
    def feedback_directory(self) -> Path:
        """Get feedback directory as Path object."""
        return Path(self.feedback_dir)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures settings are loaded only once,
    improving performance and ensuring consistency.
    """
    return Settings()
