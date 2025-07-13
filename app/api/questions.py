from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional   
import logging
from app.database import get_db, Question, Corpus
from app.models.schemas import QuestionCreate, Question as QuestionSchema, GenerateQuestionsRequest, GenerateQuestionsByTopicRequest
from app.services.question_generator import QuestionGeneratorService
from app.api.corpus import get_corpus_content
from app.services.llm_service import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post("/", response_model=QuestionSchema)
async def create_question(
    question: QuestionCreate,
    db: Session = Depends(get_db)
):
    """Create a new question"""
    logger.info(f"Creating new question for corpus_id: {question.corpus_id}")
    try:
        # Verify corpus exists
        corpus = db.query(Corpus).filter(Corpus.id == question.corpus_id).first()
        if not corpus:
            logger.error(f"Corpus not found with id: {question.corpus_id}")
            raise HTTPException(status_code=404, detail="Corpus not found")
        
        logger.debug(f"Found corpus: {corpus.name}")
        
        db_question = Question(
            corpus_id=question.corpus_id,
            question_text=question.question_text,
            reference_answer=question.reference_answer,
            generated_by=question.generated_by
        )
        
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        logger.info(f"Successfully created question with id: {db_question.id}")
        return db_question
    
    except Exception as e:
        logger.error(f"Error creating question: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[QuestionSchema])
async def list_questions(
    corpus_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """List questions, optionally filtered by corpus"""
    logger.info(f"Listing questions, corpus_id filter: {corpus_id}")
    query = db.query(Question)
    if corpus_id:
        query = query.filter(Question.corpus_id == corpus_id)
    
    questions = query.all()
    logger.info(f"Found {len(questions)} questions")
    return questions


@router.get("/{question_id}", response_model=QuestionSchema)
async def get_question(question_id: int, db: Session = Depends(get_db)):
    """Get a specific question"""
    logger.info(f"Getting question with id: {question_id}")
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        logger.error(f"Question not found with id: {question_id}")
        raise HTTPException(status_code=404, detail="Question not found")
    return question


@router.delete("/{question_id}")
async def delete_question(question_id: int, db: Session = Depends(get_db)):
    """Delete a question"""
    logger.info(f"Deleting question with id: {question_id}")
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        logger.error(f"Question not found with id: {question_id}")
        raise HTTPException(status_code=404, detail="Question not found")
    
    try:
        db.delete(question)
        db.commit()
        logger.info(f"Successfully deleted question with id: {question_id}")
        return {"message": "Question deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting question: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_questions(
    request: GenerateQuestionsRequest,
    db: Session = Depends(get_db)
):
    """Generate questions for a corpus using AI"""
    logger.info(f"Starting question generation for corpus_id: {request.corpus_id}")
    logger.info(f"Request parameters: model_provider={request.model_provider}, model_name={request.model_name}, num_questions={request.num_questions}")
    
    try:
        # Verify corpus exists
        logger.debug(f"Verifying corpus exists with id: {request.corpus_id}")
        corpus = db.query(Corpus).filter(Corpus.id == request.corpus_id).first()
        if not corpus:
            logger.error(f"Corpus not found with id: {request.corpus_id}")
            raise HTTPException(status_code=404, detail="Corpus not found")
        
        logger.info(f"Found corpus: {corpus.name}")
        
        # Get corpus content
        logger.debug("Fetching corpus content")
        corpus_content_response = await get_corpus_content(request.corpus_id, db)
        corpus_text = corpus_content_response["content"]
        logger.info(f"Retrieved corpus content, length: {len(corpus_text)} characters")
        
        # Initialize question generator
        logger.debug(f"Initializing QuestionGeneratorService with provider={request.model_provider}, model={request.model_name}")
        generator = QuestionGeneratorService(
            llm_provider=request.model_provider,
            llm_model=request.model_name
        )
        logger.info("QuestionGeneratorService initialized successfully")
        
        # Generate questions
        logger.info(f"Starting question generation for {request.num_questions} questions")
        questions = await generator.generate_questions_from_corpus(
            db, request.corpus_id, corpus_text, request.num_questions
        )
        
        logger.info(f"Successfully generated {len(questions)} questions")
        return {
            "message": f"Generated {len(questions)} questions successfully",
            "questions": [
                {
                    "id": q.id,
                    "question_text": q.question_text,
                    "reference_answer": q.reference_answer,
                    "generated_by": q.generated_by
                }
                for q in questions
            ]
        }
    
    except Exception as e:
        logger.error(f"Error in generate_questions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate-by-topic")
async def generate_questions_by_topic(
    request: GenerateQuestionsByTopicRequest,
    db: Session = Depends(get_db)
):
    """Generate questions focused on specific topics"""
    logger.info(f"Starting topic-based question generation for corpus_id: {request.corpus_id}")
    logger.info(f"Topics: {request.topics}, questions_per_topic: {request.questions_per_topic}")
    
    try:
        # Verify corpus exists
        logger.debug(f"Verifying corpus exists with id: {request.corpus_id}")
        corpus = db.query(Corpus).filter(Corpus.id == request.corpus_id).first()
        if not corpus:
            logger.error(f"Corpus not found with id: {request.corpus_id}")
            raise HTTPException(status_code=404, detail="Corpus not found")
        
        logger.info(f"Found corpus: {corpus.name}")
        
        # Get corpus content
        logger.debug("Fetching corpus content")
        corpus_content_response = await get_corpus_content(request.corpus_id, db)
        corpus_text = corpus_content_response["content"]
        logger.info(f"Retrieved corpus content, length: {len(corpus_text)} characters")
        
        # Initialize question generator
        logger.debug(f"Initializing QuestionGeneratorService with provider={request.model_provider}, model={request.model_name}")
        generator = QuestionGeneratorService(
            llm_provider=request.model_provider,
            llm_model=request.model_name
        )
        logger.info("QuestionGeneratorService initialized successfully")
        
        # Generate questions by topic
        logger.info(f"Starting topic-based question generation for {len(request.topics)} topics")
        qa_pairs = await generator.generate_questions_by_topic(
            corpus_text, request.topics, request.questions_per_topic
        )
        logger.info(f"Generated {len(qa_pairs)} question-answer pairs")
        
        # Save questions to database
        logger.debug("Saving questions to database")
        saved_questions = await generator.save_questions_to_database(
            db, request.corpus_id, qa_pairs
        )
        logger.info(f"Successfully saved {len(saved_questions)} questions to database")
        
        return {
            "message": f"Generated {len(saved_questions)} questions for {len(request.topics)} topics",
            "topics": request.topics,
            "questions": [
                {
                    "id": q.id,
                    "question_text": q.question_text,
                    "reference_answer": q.reference_answer,
                    "generated_by": q.generated_by
                }
                for q in saved_questions
            ]
        }
    
    except Exception as e:
        logger.error(f"Error in generate_questions_by_topic: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/corpus/{corpus_id}/count")
async def get_question_count(corpus_id: int, db: Session = Depends(get_db)):
    """Get the number of questions for a corpus"""
    logger.info(f"Getting question count for corpus_id: {corpus_id}")
    count = db.query(Question).filter(Question.corpus_id == corpus_id).count()
    logger.info(f"Found {count} questions for corpus {corpus_id}")
    return {"corpus_id": corpus_id, "question_count": count}


@router.get("/corpus/{corpus_id}/types")
async def get_question_types(corpus_id: int, db: Session = Depends(get_db)):
    """Get statistics about question types for a corpus"""
    logger.info(f"Getting question type statistics for corpus_id: {corpus_id}")
    questions = db.query(Question).filter(Question.corpus_id == corpus_id).all()
    
    # Count by generation method
    manual_count = sum(1 for q in questions if str(q.generated_by) == "manual")
    ai_count = sum(1 for q in questions if str(q.generated_by) == "ai")
    
    logger.info(f"Corpus {corpus_id}: {len(questions)} total questions, {manual_count} manual, {ai_count} AI-generated")
    return {
        "corpus_id": corpus_id,
        "total_questions": len(questions),
        "manual_questions": manual_count,
        "ai_generated_questions": ai_count
    } 