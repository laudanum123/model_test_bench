from datetime import datetime
from typing import cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.corpus import get_corpus_content
from app.database import (
    Corpus,
    EvaluationResult,
    EvaluationRun,
    ModelCatalogue,
    Question,
    get_db,
)
from app.models.schemas import EvaluationResult as EvaluationResultSchema
from app.models.schemas import EvaluationRun as EvaluationRunSchema
from app.models.schemas import (
    EvaluationRunCreate,
    LLMProvider,
    RetrievalStrategy,
)
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import LLMServiceFactory
from app.services.retrieval_service import RetrievalServiceFactory

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/models")
async def get_available_models():
    """Get available models for different providers"""

    import logging
    logger = logging.getLogger("get_available_models")
    logger.info("Called get_available_models endpoint")

    from app.database import ModelCatalogue, get_db

    # Get database session
    db = next(get_db())
    logger.debug("Database session created")

    try:
        # Get models from catalogue
        logger.debug("Querying embedding models from catalogue")
        embedding_models = db.query(ModelCatalogue).filter(
            ModelCatalogue.model_type == "embedding",
            ModelCatalogue.is_active == 1
        ).all()
        logger.debug(f"Found {len(embedding_models)} embedding models: {[m.huggingface_name for m in embedding_models]}")

        logger.debug("Querying reranker models from catalogue")
        reranker_models = db.query(ModelCatalogue).filter(
            ModelCatalogue.model_type == "reranker",
            ModelCatalogue.is_active == 1
        ).all()
        logger.debug(f"Found {len(reranker_models)} reranker models: {[m.huggingface_name for m in reranker_models]}")

        # Format catalogue models
        catalogue_embedding_models = [model.huggingface_name for model in embedding_models]
        catalogue_reranker_models = [model.huggingface_name for model in reranker_models]
        logger.debug(f"catalogue_embedding_models: {catalogue_embedding_models}")
        logger.debug(f"catalogue_reranker_models: {catalogue_reranker_models}")

        result = {
            "llm_providers": {
                "openai": [
                    "gpt-4.1",
                ],
                "transformers": [
                    "Empty"
                ]
            },
            "embedding_providers": {
                "openai": [
                    "text-embedding-ada-002"
                ],
                "sentence_transformers": catalogue_embedding_models
            },
            "reranker_models": catalogue_reranker_models,
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
        logger.info(f"Returning model catalogue: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in get_available_models: {e!s}", exc_info=True)
        raise
    finally:
        db.close()
        logger.debug("Database session closed")


@router.post("/", response_model=EvaluationRunSchema)
async def create_evaluation_run(
    evaluation: EvaluationRunCreate,
    db: Session = Depends(get_db)
):
    """Create a new evaluation run"""
    try:
        # Verify corpus exists
        corpus = db.query(Corpus).filter(Corpus.id == evaluation.corpus_id).first()
        if not corpus:
            raise HTTPException(status_code=404, detail="Corpus not found")

        # Create evaluation run
        db_evaluation = EvaluationRun(
            name=evaluation.name,
            description=evaluation.description,
            corpus_id=evaluation.corpus_id,
            llm_provider=evaluation.stack_config.llm_provider.value,
            llm_model=evaluation.stack_config.llm_model,
            embedding_provider=evaluation.stack_config.embedding_provider.value,
            embedding_model=evaluation.stack_config.embedding_model,
            reranker_provider=evaluation.stack_config.reranker_provider,
            reranker_model=evaluation.stack_config.reranker_model,
            vector_store=evaluation.stack_config.vector_store.value,
            retrieval_strategy=evaluation.stack_config.retrieval_strategy.value
        )

        db.add(db_evaluation)
        db.commit()
        db.refresh(db_evaluation)

        return db_evaluation

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/", response_model=list[EvaluationRunSchema])
async def list_evaluation_runs(db: Session = Depends(get_db)):
    """List all evaluation runs"""
    evaluations = db.query(EvaluationRun).all()
    return evaluations


@router.get("/{evaluation_id}", response_model=EvaluationRunSchema)
async def get_evaluation_run(evaluation_id: int, db: Session = Depends(get_db)):
    """Get a specific evaluation run"""
    evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return evaluation


@router.post("/{evaluation_id}/run")
async def run_evaluation(
    evaluation_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Run an evaluation in the background"""
    evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    if str(evaluation.status) == "running":
        raise HTTPException(status_code=400, detail="Evaluation is already running")

    # Start evaluation in background
    background_tasks.add_task(run_evaluation_task, evaluation_id, db)

    return {"message": "Evaluation started", "evaluation_id": evaluation_id}


@router.post("/{evaluation_id}/cancel")
async def cancel_evaluation(evaluation_id: int, db: Session = Depends(get_db)):
    """Cancel a running evaluation"""
    evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    if str(evaluation.status) != "running":
        raise HTTPException(status_code=400, detail="Only running evaluations can be cancelled")

    # Update status to cancelled
    db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).update({
        "status": "cancelled",
        "completed_at": datetime.utcnow()
    })
    db.commit()

    return {"message": "Evaluation cancelled", "evaluation_id": evaluation_id}


@router.delete("/{evaluation_id}")
async def delete_evaluation(evaluation_id: int, db: Session = Depends(get_db)):
    """Delete an evaluation run and all its results"""
    evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    if str(evaluation.status) == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running evaluation. Cancel it first.")

    try:
        # Delete all evaluation results first (due to foreign key constraint)
        db.query(EvaluationResult).filter(
            EvaluationResult.evaluation_run_id == evaluation_id
        ).delete()

        # Delete the evaluation run
        db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).delete()
        db.commit()

        return {"message": "Evaluation deleted successfully", "evaluation_id": evaluation_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting evaluation: {e!s}") from e


async def run_evaluation_task(evaluation_id: int, db: Session):
    """Background task to run the evaluation"""
    try:
        # Get evaluation run
        evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
        if not evaluation:
            return

        # Update status to running
        db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).update({"status": "running"})
        db.commit()

        # Get corpus content
        corpus_content_response = await get_corpus_content(Corpus(id=evaluation.corpus_id), db)
        corpus_text = corpus_content_response["content"]

        # Get questions for this corpus
        questions = db.query(Question).filter(Question.corpus_id == evaluation.corpus_id).all()
        if not questions:
            raise Exception("No questions found for this corpus")

        # Get model information from catalogue

        # Look up embedding model
        embedding_model_info = db.query(ModelCatalogue).filter(
            ModelCatalogue.huggingface_name == evaluation.embedding_model,
            ModelCatalogue.model_type == "embedding",
            ModelCatalogue.is_active == 1
        ).first()

        # Look up reranker model if specified
        reranker_model_info = None
        if evaluation.reranker_model is not None:
            reranker_model_info = db.query(ModelCatalogue).filter(
                ModelCatalogue.huggingface_name == evaluation.reranker_model,
                ModelCatalogue.model_type == "reranker",
                ModelCatalogue.is_active == 1
            ).first()

        # Initialize services
        llm_service = LLMServiceFactory.create_service(
            LLMProvider(evaluation.llm_provider), str(evaluation.llm_model)
        )

        # Use local paths from catalogue if available, otherwise fall back to HuggingFace names
        embedding_model_path = embedding_model_info.local_path if embedding_model_info else evaluation.embedding_model
        reranker_model_path = reranker_model_info.local_path if reranker_model_info else evaluation.reranker_model

        retrieval_service = RetrievalServiceFactory.create_service(
            RetrievalStrategy(evaluation.retrieval_strategy),
            embedding_provider=evaluation.embedding_provider,
            embedding_model=embedding_model_path,
            vector_store=evaluation.vector_store,
            reranker_model=reranker_model_path
        )

        evaluation_service = EvaluationService()

        # Add documents to retrieval system
        # Split corpus into chunks
        chunks = split_text_into_chunks(corpus_text, chunk_size=1000, overlap=200)
        await retrieval_service.add_documents(chunks)

        # Process each question
        total_questions = len(questions)
        correct_answers = 0
        total_relevance_score = 0
        total_retrieval_score = 0

        for _, question in enumerate(questions):
            # Check if evaluation has been cancelled
            current_evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
            if current_evaluation and str(current_evaluation.status) == "cancelled":
                print(f"Evaluation {evaluation_id} was cancelled")
                return

            try:
                # Retrieve relevant documents
                retrieved_docs = await retrieval_service.retrieve(str(question.question_text), top_k=5)
                retrieved_chunks = [doc["text"] for doc in retrieved_docs]

                # Generate answer
                generated_answer = await llm_service.generate_with_context(
                    str(question.question_text), retrieved_chunks
                )

                # Evaluate the result
                scores = await evaluation_service.evaluate_single_result(
                    str(question.question_text),
                    str(question.reference_answer),
                    generated_answer,
                    retrieved_chunks
                )

                # Save evaluation result
                await evaluation_service.save_evaluation_result(
                    db, evaluation_id, Question(id=cast("int", question.id)), generated_answer, retrieved_docs, scores
                )

                # Update counters
                if scores["answer_relevance_score"] > 0.7:  # Threshold for "correct"
                    correct_answers += 1

                total_relevance_score += scores["answer_relevance_score"]
                total_retrieval_score += scores["retrieval_relevance_score"]

            except Exception as e:
                print(f"Error processing question {question.id}: {e!s}")
                continue

        # Final check for cancellation before updating final results
        current_evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
        if current_evaluation and str(current_evaluation.status) == "cancelled":
            print(f"Evaluation {evaluation_id} was cancelled before finalizing results")
            return

        # Update evaluation run with results
        db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).update({
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": correct_answers / total_questions if total_questions > 0 else 0,
            "average_relevance_score": total_relevance_score / total_questions if total_questions > 0 else 0,
            "average_retrieval_score": total_retrieval_score / total_questions if total_questions > 0 else 0,
            "completed_at": datetime.utcnow(),
            "status": "completed"
        })

        db.commit()

    except Exception as e:
        # Update status to failed
        db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).update({"status": "failed"})
        db.commit()
        print(f"Evaluation {evaluation_id} failed: {e!s}")


@router.get("/{evaluation_id}/results", response_model=list[EvaluationResultSchema])
async def get_evaluation_results(evaluation_id: int, db: Session = Depends(get_db)):
    """Get results for a specific evaluation run"""
    results = db.query(EvaluationResult).filter(
        EvaluationResult.evaluation_run_id == evaluation_id
    ).all()
    return results


@router.get("/{evaluation_id}/summary")
async def get_evaluation_summary(evaluation_id: int, db: Session = Depends(get_db)):
    """Get a summary of evaluation results"""
    evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = db.query(EvaluationResult).filter(
        EvaluationResult.evaluation_run_id == evaluation_id
    ).all()

    if not results:
        return {
            "evaluation_id": evaluation_id,
            "status": evaluation.status,
            "message": "No results available yet"
        }

    # Calculate additional metrics
    avg_answer_relevance = sum(r.answer_relevance_score or 0 for r in results) / len(results)
    avg_retrieval_relevance = sum(r.retrieval_relevance_score or 0 for r in results) / len(results)
    avg_chunk_overlap = sum(r.chunk_overlap_score or 0 for r in results) / len(results)

    return {
        "evaluation_id": evaluation_id,
        "status": evaluation.status,
        "total_questions": evaluation.total_questions,
        "correct_answers": evaluation.correct_answers,
        "accuracy": evaluation.accuracy,
        "average_relevance_score": evaluation.average_relevance_score,
        "average_retrieval_score": evaluation.average_retrieval_score,
        "detailed_metrics": {
            "avg_answer_relevance": avg_answer_relevance,
            "avg_retrieval_relevance": avg_retrieval_relevance,
            "avg_chunk_overlap": avg_chunk_overlap
        },
        "completed_at": evaluation.completed_at
    }


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start, end - 100), -1):
                if text[i] in ".!?":
                    end = i + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks
