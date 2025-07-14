from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import settings

Base = declarative_base()

class Corpus(Base):
    __tablename__ = "corpora"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    source = Column(String)  # "huggingface", "upload", "custom"
    source_config = Column(JSON)  # Configuration for the source
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelCatalogue(Base):
    __tablename__ = "model_catalogue"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    model_type = Column(String, nullable=False)  # "embedding", "reranker"
    provider = Column(String, nullable=False)  # "sentence_transformers", "transformers", "openai"
    huggingface_name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    model_info = Column(JSON)  # Additional model information (dimensions, etc.)
    local_path = Column(String)  # Local path where model is stored
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    corpus_id = Column(Integer, nullable=False)
    question_text = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=True)
    generated_by = Column(String)  # "manual", "ai"
    created_at = Column(DateTime, default=datetime.utcnow)

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    corpus_id = Column(Integer, nullable=False)

    # Stack Configuration
    llm_provider = Column(String, nullable=False)  # "openai", "transformers"
    llm_model = Column(String, nullable=False)
    embedding_provider = Column(String, nullable=False)  # "openai", "sentence_transformers"
    embedding_model = Column(String, nullable=False)
    reranker_provider = Column(String)  # "transformers", None
    reranker_model = Column(String)
    vector_store = Column(String, nullable=False)  # "chroma", "faiss"
    retrieval_strategy = Column(String, nullable=False)  # "semantic", "hybrid", "bm25"

    # Results
    total_questions = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    average_relevance_score = Column(Float, default=0.0)
    average_retrieval_score = Column(Float, default=0.0)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String, default="running")  # "running", "completed", "failed"

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_run_id = Column(Integer, nullable=False)
    question_id = Column(Integer, nullable=False)

    # Generated Answer
    generated_answer = Column(Text, nullable=False)

    # Retrieved Chunks
    retrieved_chunks = Column(JSON)  # List of chunk texts and metadata

    # Evaluation Scores
    answer_relevance_score = Column(Float)  # How well the answer matches reference
    retrieval_relevance_score = Column(Float)  # How relevant retrieved chunks are
    chunk_overlap_score = Column(Float)  # How much relevant info is in chunks

    # Detailed Evaluation
    evaluation_details = Column(JSON)  # Detailed breakdown of evaluation

    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)
