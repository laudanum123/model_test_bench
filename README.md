# Model Test Bench

A comprehensive web application for evaluating different LLM provider stacks with various embedding models, rerankers, and vector stores.

## Features

- **Multi-Provider LLM Support**: OpenAI API and local Transformers models
- **Flexible Embedding Models**: OpenAI embeddings and Sentence Transformers
- **Reranker Models**: Support for various reranker models via Transformers
- **Vector Store Options**: ChromaDB, FAISS, and more
- **Retrieval Strategies**: Hybrid, semantic, BM25, and more
- **Evaluation Corpus**: Support for HuggingFace datasets and custom corpora
- **Question Generation**: Automatic synthetic Q&A dataset generation
- **Answer Evaluation**: AI-powered judge models for answer quality assessment
- **Result Storage**: Persistent storage of all test runs and evaluations
- **Model Catalogue**: Manage and download embedding and reranker models from HuggingFace

## Model Catalogue

The Model Catalogue feature allows you to:

- **Download Models**: Add embedding and reranker models from HuggingFace Hub
- **Local Storage**: Models are downloaded and stored locally for faster access
- **Model Information**: View detailed information about each model including dimensions, vocabulary size, etc.
- **Model Management**: Delete models you no longer need
- **Automatic Integration**: Downloaded models are automatically available in evaluations

### Adding Models

1. Navigate to the **Model Catalogue** page
2. Click **Add Model**
3. Select the model type (Embedding or Reranker)
4. Enter the HuggingFace model name (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
5. Optionally provide a custom display name and description
6. Click **Download Model**

The model will be downloaded in the background and added to your catalogue. Once downloaded, it will be available for use in evaluations.

### Model Types

- **Embedding Models**: Used for generating text embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- **Reranker Models**: Used for reranking retrieved documents (e.g., `BAAI/bge-reranker-v2-m3`)

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the application**:
   ```bash
   uv run python -m app.main
   ```

4. **Access the web interface**:
   Open http://localhost:8000 in your browser

5. **Add some models**:
   - Go to **Model Catalogue**
   - Add embedding models like `sentence-transformers/all-MiniLM-L6-v2`
   - Add reranker models like `BAAI/bge-reranker-v2-m3`

## Project Structure

```
model_test_bench/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration settings
│   ├── database.py             # Database models and setup
│   ├── models/                 # Pydantic models
│   ├── services/               # Business logic services
│   │   ├── llm_service.py      # LLM provider abstraction
│   │   ├── embedding_service.py # Embedding model abstraction
│   │   ├── reranker_service.py # Reranker model abstraction
│   │   ├── vector_store_service.py # Vector store abstraction
│   │   ├── retrieval_service.py # Retrieval strategy implementation
│   │   ├── evaluation_service.py # Evaluation logic
│   │   └── question_generator.py # Synthetic Q&A generation
│   ├── api/                    # API routes
│   └── templates/              # HTML templates
├── tests/                      # Test files
└── data/                       # Data storage
```

## Usage

1. **Upload or select a corpus**: Choose from HuggingFace datasets or upload your own text
2. **Configure your stack**: Select LLM provider, embedding model, reranker, and vector store
3. **Generate or create questions**: Use AI to generate questions or create custom ones
4. **Run evaluation**: Execute the full pipeline and get detailed results
5. **Analyze results**: View performance metrics and compare different configurations

## Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_huggingface_token
DATABASE_URL=sqlite:///./data/evaluation.db
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

