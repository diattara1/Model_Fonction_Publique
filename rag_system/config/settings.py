import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model configuration
    EMBED_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    LLM_MODEL_NAME: str = "Qwen/Qwen3-14B-AWQ"
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # Document processing
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 250
    TOP_K: int = 5
    MAX_STEPS: int = 3  # AJOUTÉ CETTE LIGNE
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Paths
    PDF_DIR: str = "data/pdf"
    INDICES_DIR: str = "data/indices"
    
    # Security
    CORS_ORIGINS: list = ["*"]

settings = Settings()