from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from app.config import settings
from app.database import create_tables
from app.api import corpus, evaluation, questions

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A comprehensive web application for evaluating different LLM provider stacks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(corpus.router)
app.include_router(evaluation.router)
app.include_router(questions.router)

# Create templates directory and mount static files
templates_dir = "app/templates"
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Mount static files
static_dir = "app/static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    # Create database tables
    create_tables()
    print(f"🚀 {settings.app_name} started successfully!")


@app.get("/")
async def root(request: Request):
    """Root endpoint - redirect to dashboard"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard")
async def dashboard(request: Request):
    """Dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_page": "dashboard"})


@app.get("/corpus")
async def corpus_page(request: Request):
    """Corpus management page"""
    return templates.TemplateResponse("corpus.html", {"request": request, "active_page": "corpus"})


@app.get("/questions")
async def questions_page(request: Request):
    """Questions page"""
    return templates.TemplateResponse("questions.html", {"request": request, "active_page": "questions"})


@app.get("/evaluation")
async def evaluation_page(request: Request):
    """Evaluation list page"""
    return templates.TemplateResponse("evaluation.html", {"request": request, "active_page": "evaluation"})


@app.get("/new-evaluation")
async def new_evaluation_page(request: Request):
    """New evaluation creation page"""
    return templates.TemplateResponse("new_evaluation.html", {"request": request, "active_page": "new-evaluation"})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "app": settings.app_name}


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    from fastapi.responses import FileResponse
    import os
    favicon_path = os.path.join("app", "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    else:
        from fastapi.responses import Response
        return Response(status_code=404)


@app.get("/api/models")
async def get_available_models():
    """Get available models for different providers"""
    return {
        "llm_providers": {
            "openai": [
                "gpt-4.1",
                "gpt-4",
                "gpt-4-turbo-preview"
            ],
            "transformers": [
                "microsoft/DialoGPT-medium",
                "gpt2",
                "EleutherAI/gpt-neo-125M"
            ]
        },
        "embedding_providers": {
            "openai": [
                "text-embedding-ada-002"
            ],
            "sentence_transformers": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5"
            ]
        },
        "reranker_models": [
            "BAAI/bge-reranker-v2-m3",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2"
        ],
        "vector_stores": [
            "chroma",
            "faiss"
        ],
        "retrieval_strategies": [
            "semantic",
            "hybrid",
            "bm25"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    ) 