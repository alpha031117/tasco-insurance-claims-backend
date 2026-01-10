# config.py
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    anthropic_api_key: str
    # Supabase configuration for pgvector RAG
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    # Voyage AI configuration for embeddings
    voyage_api_key: Optional[str] = None
    embedding_dimension: int = 1024  # Voyage AI default dimension (1024)
    embedding_model: str = "voyage-2"  # Voyage AI model (voyage-2, voyage-lite-02, voyage-large-2, etc.)

    class Config:
        env_file = ".env"

settings = Settings()
