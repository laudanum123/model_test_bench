from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Database
    database_url: str = "sqlite:///./data/evaluation.db"
    
    # Application
    app_name: str = "Model Test Bench"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model Settings
    default_llm_model: str = "gpt-3.5-turbo"
    default_embedding_model: str = "text-embedding-ada-002"
    default_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    
    # Vector Store Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Evaluation Settings
    max_questions_per_corpus: int = 50
    evaluation_timeout: int = 300  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Ensure data directory exists
os.makedirs("data", exist_ok=True) 