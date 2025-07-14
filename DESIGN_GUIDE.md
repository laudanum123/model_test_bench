I'll analyze the codebase to understand the application structure and create a comprehensive description. Let me explore the project to gather all the relevant information.

Read file: README.md
Read file: pyproject.toml
Read file: app/config.py
Read file: app/database.py
Read file: app/models/schemas.py
Ran tool
Read file: app/typing_helpers/vector_store_protocols.py
Read file: TEMPLATE_STRUCTURE.md
Based on my analysis of the codebase, I'll create a comprehensive description of the Model Test Bench application, its tech stack, and design choices.

# Model Test Bench - Application Description

## Overview

The **Model Test Bench** is a comprehensive web application designed for evaluating and comparing different LLM (Large Language Model) provider stacks. It provides a systematic approach to testing various combinations of language models, embedding models, rerankers, and vector stores to determine the optimal configuration for specific use cases.

## Core Purpose

The application addresses the challenge of selecting the best AI/ML stack for retrieval-augmented generation (RAG) systems by providing:

- **Systematic Evaluation**: Automated testing of different model combinations
- **Performance Metrics**: Quantitative assessment of answer quality and retrieval accuracy
- **Comparative Analysis**: Side-by-side comparison of different configurations
- **Reproducible Results**: Persistent storage of all test runs and evaluations

## Key Features

### 1. **Multi-Provider LLM Support**
- **OpenAI API**: GPT-4, GPT-4 Turbo, and other OpenAI models
- **Local Transformers**: Microsoft DialoGPT, GPT-2, GPT-Neo models
- **Provider Abstraction**: Unified interface for different LLM providers

### 2. **Flexible Embedding Models**
- **OpenAI Embeddings**: text-embedding-ada-002 and other OpenAI embedding models
- **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2, BAAI/bge-small-en-v1.5
- **Cross-Provider Compatibility**: Mix and match embedding providers with different LLMs

### 3. **Advanced Reranking**
- **Multiple Reranker Models**: BAAI/bge-reranker-v2-m3, cross-encoder models
- **Optional Integration**: Rerankers can be enabled/disabled based on requirements
- **Performance Optimization**: Improve retrieval quality through reranking

### 4. **Vector Store Options**
- **ChromaDB**: Persistent, feature-rich vector database
- **FAISS**: High-performance similarity search library
- **Extensible Architecture**: Easy addition of new vector stores

### 5. **Retrieval Strategies**
- **Semantic Search**: Pure vector similarity search
- **Hybrid Search**: Combination of semantic and keyword-based search
- **BM25**: Traditional keyword-based retrieval

### 6. **Corpus Management**
- **HuggingFace Integration**: Direct loading of datasets from HuggingFace Hub
- **File Upload**: Support for custom text files and documents
- **Dataset Processing**: Automatic chunking and preprocessing

### 7. **Question Generation**
- **AI-Powered Generation**: Automatic creation of synthetic Q&A datasets
- **Topic-Based Generation**: Generate questions focused on specific topics
- **Manual Creation**: Support for custom question creation

### 8. **Evaluation Framework**
- **AI Judge Models**: Automated assessment of answer quality
- **Multiple Metrics**: Relevance scores, retrieval accuracy, chunk overlap
- **Detailed Analysis**: Comprehensive evaluation breakdowns

## Tech Stack

### Backend Framework
- **FastAPI**: Modern, fast web framework for building APIs with Python
- **Python 3.12+**: Latest Python version for optimal performance
- **Uvicorn**: ASGI server for production deployment

### Database & ORM
- **SQLAlchemy 2.0**: Modern Python ORM with async support
- **SQLite**: Lightweight database for development and small deployments
- **Alembic**: Database migration management

### AI/ML Libraries
- **Transformers**: HuggingFace library for state-of-the-art NLP models
- **PyTorch**: Deep learning framework for local model inference
- **Sentence Transformers**: Specialized library for sentence embeddings
- **OpenAI**: Official Python client for OpenAI API integration

### Vector Search & Storage
- **ChromaDB**: Open-source embedding database
- **FAISS**: Facebook AI Similarity Search library
- **NumPy**: Numerical computing for vector operations

### Data Processing
- **Pandas**: Data manipulation and analysis
- **Datasets**: HuggingFace datasets library
- **Scikit-learn**: Machine learning utilities
- **Rank-BM25**: Traditional information retrieval

### Frontend
- **Jinja2**: Server-side templating engine
- **Bootstrap 5**: Modern CSS framework for responsive design
- **Font Awesome**: Icon library
- **Vanilla JavaScript**: Lightweight client-side interactions

### Development Tools
- **UV**: Fast Python package manager and project management
- **Pytest**: Testing framework with async support
- **Black**: Code formatter
- **Ruff**: Fast Python linter and formatter
- **Pydantic**: Data validation and settings management

## Architecture & Design Choices

### 1. **Service-Oriented Architecture**
The application follows a clean service-oriented architecture with clear separation of concerns:

```
app/
├── services/           # Business logic layer
│   ├── llm_service.py      # LLM provider abstraction
│   ├── embedding_service.py # Embedding model abstraction
│   ├── reranker_service.py # Reranker abstraction
│   ├── vector_store_service.py # Vector store abstraction
│   ├── retrieval_service.py # Retrieval strategy implementation
│   ├── evaluation_service.py # Evaluation logic
│   └── question_generator.py # Q&A generation
├── api/               # API endpoints
├── models/            # Pydantic schemas
└── templates/         # Frontend templates
```

### 2. **Provider Abstraction Pattern**
Each service implements a provider abstraction pattern, allowing easy switching between different providers:

```python
# Example: LLM Service
class LLMService:
    def __init__(self, provider: str, model: str):
        self.provider = self._get_provider(provider, model)
    
    def _get_provider(self, provider: str, model: str):
        if provider == "openai":
            return OpenAIProvider(model)
        elif provider == "transformers":
            return TransformersProvider(model)
```

### 3. **Protocol-Based Type Safety**
Uses Python protocols for runtime type checking and interface definition:

```python
@runtime_checkable
class SupportsUpdateEmbeddings(Protocol):
    async def update_embeddings(self, embeddings: list[list[float]]) -> None: ...
```

### 4. **Modular Template Structure**
Replaced single-page application with modular Jinja2 templates:
- **Base Template**: Common layout, navigation, and utilities
- **Page-Specific Templates**: Focused functionality for each feature
- **Progressive Enhancement**: Works without JavaScript, enhanced with JS

### 5. **Configuration Management**
Centralized configuration using Pydantic Settings:
- Environment variable support
- Type validation
- Default values
- Development/production mode switching

### 6. **Database Design**
Well-structured SQLAlchemy models with proper relationships:
- **Corpora**: Document collections and metadata
- **Questions**: Q&A pairs for evaluation
- **EvaluationRuns**: Test configurations and results
- **EvaluationResults**: Detailed per-question results

### 7. **Async/Await Pattern**
Leverages Python's async capabilities for:
- Non-blocking I/O operations
- Concurrent processing
- Better resource utilization

## Development Conventions

### 1. **Code Quality**
- **Ruff**: Fast linting and formatting
- **Black**: Consistent code formatting
- **Type Hints**: Comprehensive type annotations
- **Docstrings**: Clear documentation

### 2. **Testing Strategy**
- **Pytest**: Testing framework with async support
- **Integration Tests**: End-to-end API testing
- **Unit Tests**: Individual component testing
- **Test Coverage**: Comprehensive coverage reporting

### 3. **Package Management**
- **UV**: Fast dependency resolution and installation
- **Lock Files**: Reproducible builds
- **Development Dependencies**: Separate dev and production dependencies

### 4. **Environment Management**
- **Virtual Environments**: Isolated Python environments
- **Environment Variables**: Secure configuration management
- **Example Files**: Clear setup instructions

## Deployment & Operations

### 1. **Development Setup**
```bash
uv sync                    # Install dependencies
cp .env.example .env      # Configure environment
uv run python -m app.main # Start development server
```

### 2. **Production Considerations**
- **Database**: SQLite for development, PostgreSQL for production
- **Static Files**: Proper serving configuration
- **Logging**: Structured logging with different levels
- **Health Checks**: Built-in health check endpoints

### 3. **Monitoring & Observability**
- **Structured Logging**: JSON-formatted logs
- **Performance Metrics**: Evaluation timing and accuracy
- **Error Handling**: Graceful error recovery

## Future Enhancements

The modular architecture enables several future improvements:

1. **Additional Providers**: Support for more LLM and embedding providers
2. **Advanced Metrics**: More sophisticated evaluation criteria
3. **Batch Processing**: Parallel evaluation of multiple configurations
4. **API Integration**: RESTful API for external integrations
5. **Real-time Updates**: WebSocket support for live evaluation progress
6. **Export Capabilities**: Results export in various formats
7. **User Management**: Multi-user support with authentication

This architecture provides a solid foundation for a comprehensive LLM evaluation platform that can adapt to new models, providers, and evaluation methodologies as the field evolves.