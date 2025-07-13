from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    OPENAI = "openai"
    TRANSFORMERS = "transformers"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class VectorStore(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"


class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    BM25 = "bm25"


class CorpusBase(BaseModel):
    name: str = Field(..., description="Name of the corpus")
    description: str | None = Field(None, description="Description of the corpus")
    source: str = Field(..., description="Source type: huggingface, upload, custom")
    source_config: dict[str, Any] = Field(..., description="Configuration for the source")


class CorpusCreate(CorpusBase):
    pass


class Corpus(CorpusBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class QuestionBase(BaseModel):
    question_text: str = Field(..., description="The question text")
    reference_answer: str | None = Field(None, description="The reference answer")
    generated_by: str = Field(..., description="How the question was generated: manual, ai")


class QuestionCreate(QuestionBase):
    corpus_id: int


class Question(QuestionBase):
    id: int
    corpus_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class StackConfiguration(BaseModel):
    llm_provider: LLMProvider
    llm_model: str
    embedding_provider: EmbeddingProvider
    embedding_model: str
    reranker_provider: str | None = None
    reranker_model: str | None = None
    vector_store: VectorStore
    retrieval_strategy: RetrievalStrategy


class EvaluationRunBase(BaseModel):
    name: str = Field(..., description="Name of the evaluation run")
    description: str | None = Field(None, description="Description of the evaluation")
    corpus_id: int
    stack_config: StackConfiguration


class EvaluationRunCreate(EvaluationRunBase):
    pass


class EvaluationRun(EvaluationRunBase):
    id: int
    total_questions: int
    correct_answers: int
    accuracy: float
    average_relevance_score: float
    average_retrieval_score: float
    created_at: datetime
    completed_at: datetime | None
    status: str

    class Config:
        from_attributes = True


class EvaluationResultBase(BaseModel):
    evaluation_run_id: int
    question_id: int
    generated_answer: str
    retrieved_chunks: list[dict[str, Any]]
    answer_relevance_score: float | None = None
    retrieval_relevance_score: float | None = None
    chunk_overlap_score: float | None = None
    evaluation_details: dict[str, Any] | None = None


class EvaluationResultCreate(EvaluationResultBase):
    pass


class EvaluationResult(EvaluationResultBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class GenerateQuestionsRequest(BaseModel):
    corpus_id: int
    num_questions: int = 5
    model_provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4.1"

class GenerateQuestionsByTopicRequest(BaseModel):
    corpus_id: int
    topics: list[str]
    questions_per_topic: int = 3
    model_provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4.1"


class QuestionGenerationRequest(BaseModel):
    corpus_id: int
    num_questions: int = Field(5, ge=1, le=50, description="Number of questions to generate")
    model_provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = Field("gpt-4.1", description="Model to use for generation")


class EvaluationRequest(BaseModel):
    evaluation_run_id: int


class EvaluationResponse(BaseModel):
    evaluation_run_id: int
    status: str
    message: str
    results: list[EvaluationResult] | None = None


class HuggingFaceCorpusRequest(BaseModel):
    name: str = Field(..., description="Name of the corpus")
    dataset_name: str = Field(..., description="HuggingFace dataset name")
    split: str = Field("train", description="Dataset split to use")
    text_column: str = Field("text", description="Column name containing the text data")
    config_name: str | None = Field(None, description="Dataset configuration name")
    description: str | None = Field(None, description="Description of the corpus")
