from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import asyncio
from datetime import datetime
from app.database import get_db, EvaluationRun, EvaluationResult, Question, Corpus
from app.models.schemas import (
    EvaluationRunCreate, EvaluationRun as EvaluationRunSchema,
    EvaluationResult as EvaluationResultSchema, EvaluationRequest, EvaluationResponse
)
from app.services.llm_service import LLMServiceFactory
from app.services.retrieval_service import RetrievalServiceFactory
from app.services.evaluation_service import EvaluationService
from app.api.corpus import get_corpus_content

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


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
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[EvaluationRunSchema])
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
    
    if evaluation.status == "running":
        raise HTTPException(status_code=400, detail="Evaluation is already running")
    
    # Start evaluation in background
    background_tasks.add_task(run_evaluation_task, evaluation_id, db)
    
    return {"message": "Evaluation started", "evaluation_id": evaluation_id}


async def run_evaluation_task(evaluation_id: int, db: Session):
    """Background task to run the evaluation"""
    try:
        # Get evaluation run
        evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
        if not evaluation:
            return
        
        # Update status to running
        evaluation.status = "running"
        db.commit()
        
        # Get corpus content
        corpus_content_response = await get_corpus_content(evaluation.corpus_id, db)
        corpus_text = corpus_content_response["content"]
        
        # Get questions for this corpus
        questions = db.query(Question).filter(Question.corpus_id == evaluation.corpus_id).all()
        if not questions:
            raise Exception("No questions found for this corpus")
        
        # Initialize services
        llm_service = LLMServiceFactory.create_service(
            evaluation.llm_provider, evaluation.llm_model
        )
        
        retrieval_service = RetrievalServiceFactory.create_service(
            evaluation.retrieval_strategy,
            embedding_provider=evaluation.embedding_provider,
            embedding_model=evaluation.embedding_model,
            vector_store=evaluation.vector_store,
            reranker_model=evaluation.reranker_model
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
        
        for question in questions:
            try:
                # Retrieve relevant documents
                retrieved_docs = await retrieval_service.retrieve(question.question_text, top_k=5)
                retrieved_chunks = [doc['text'] for doc in retrieved_docs]
                
                # Generate answer
                generated_answer = await llm_service.generate_with_context(
                    question.question_text, retrieved_chunks
                )
                
                # Evaluate the result
                scores = await evaluation_service.evaluate_single_result(
                    question.question_text,
                    question.reference_answer,
                    generated_answer,
                    retrieved_chunks
                )
                
                # Save evaluation result
                await evaluation_service.save_evaluation_result(
                    db, evaluation_id, question.id, generated_answer, retrieved_docs, scores
                )
                
                # Update counters
                if scores['answer_relevance_score'] > 0.7:  # Threshold for "correct"
                    correct_answers += 1
                
                total_relevance_score += scores['answer_relevance_score']
                total_retrieval_score += scores['retrieval_relevance_score']
                
            except Exception as e:
                print(f"Error processing question {question.id}: {str(e)}")
                continue
        
        # Update evaluation run with results
        evaluation.total_questions = total_questions
        evaluation.correct_answers = correct_answers
        evaluation.accuracy = correct_answers / total_questions if total_questions > 0 else 0
        evaluation.average_relevance_score = total_relevance_score / total_questions if total_questions > 0 else 0
        evaluation.average_retrieval_score = total_retrieval_score / total_questions if total_questions > 0 else 0
        evaluation.completed_at = datetime.utcnow()
        evaluation.status = "completed"
        
        db.commit()
        
    except Exception as e:
        # Update status to failed
        evaluation = db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
        if evaluation:
            evaluation.status = "failed"
            db.commit()
        print(f"Evaluation {evaluation_id} failed: {str(e)}")


@router.get("/{evaluation_id}/results", response_model=List[EvaluationResultSchema])
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


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
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
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks 