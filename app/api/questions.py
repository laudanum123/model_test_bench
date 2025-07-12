from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db, Question, Corpus
from app.models.schemas import QuestionCreate, Question as QuestionSchema
from app.services.question_generator import QuestionGeneratorService
from app.api.corpus import get_corpus_content

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post("/", response_model=QuestionSchema)
async def create_question(
    question: QuestionCreate,
    db: Session = Depends(get_db)
):
    """Create a new question"""
    try:
        # Verify corpus exists
        corpus = db.query(Corpus).filter(Corpus.id == question.corpus_id).first()
        if not corpus:
            raise HTTPException(status_code=404, detail="Corpus not found")
        
        db_question = Question(
            corpus_id=question.corpus_id,
            question_text=question.question_text,
            reference_answer=question.reference_answer,
            generated_by=question.generated_by
        )
        
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        return db_question
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[QuestionSchema])
async def list_questions(
    corpus_id: int = None,
    db: Session = Depends(get_db)
):
    """List questions, optionally filtered by corpus"""
    query = db.query(Question)
    if corpus_id:
        query = query.filter(Question.corpus_id == corpus_id)
    
    questions = query.all()
    return questions


@router.get("/{question_id}", response_model=QuestionSchema)
async def get_question(question_id: int, db: Session = Depends(get_db)):
    """Get a specific question"""
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return question


@router.delete("/{question_id}")
async def delete_question(question_id: int, db: Session = Depends(get_db)):
    """Delete a question"""
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    try:
        db.delete(question)
        db.commit()
        return {"message": "Question deleted successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_questions(
    corpus_id: int,
    num_questions: int = 5,
    model_provider: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    db: Session = Depends(get_db)
):
    """Generate questions for a corpus using AI"""
    try:
        # Verify corpus exists
        corpus = db.query(Corpus).filter(Corpus.id == corpus_id).first()
        if not corpus:
            raise HTTPException(status_code=404, detail="Corpus not found")
        
        # Get corpus content
        corpus_content_response = await get_corpus_content(corpus_id, db)
        corpus_text = corpus_content_response["content"]
        
        # Initialize question generator
        generator = QuestionGeneratorService(
            llm_provider=model_provider,
            llm_model=model_name
        )
        
        # Generate questions
        questions = await generator.generate_questions_from_corpus(
            db, corpus_id, corpus_text, num_questions
        )
        
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
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate-by-topic")
async def generate_questions_by_topic(
    corpus_id: int,
    topics: List[str],
    questions_per_topic: int = 3,
    model_provider: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    db: Session = Depends(get_db)
):
    """Generate questions focused on specific topics"""
    try:
        # Verify corpus exists
        corpus = db.query(Corpus).filter(Corpus.id == corpus_id).first()
        if not corpus:
            raise HTTPException(status_code=404, detail="Corpus not found")
        
        # Get corpus content
        corpus_content_response = await get_corpus_content(corpus_id, db)
        corpus_text = corpus_content_response["content"]
        
        # Initialize question generator
        generator = QuestionGeneratorService(
            llm_provider=model_provider,
            llm_model=model_name
        )
        
        # Generate questions by topic
        qa_pairs = await generator.generate_questions_by_topic(
            corpus_text, topics, questions_per_topic
        )
        
        # Save questions to database
        saved_questions = await generator.save_questions_to_database(
            db, corpus_id, qa_pairs
        )
        
        return {
            "message": f"Generated {len(saved_questions)} questions for {len(topics)} topics",
            "topics": topics,
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
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/corpus/{corpus_id}/count")
async def get_question_count(corpus_id: int, db: Session = Depends(get_db)):
    """Get the number of questions for a corpus"""
    count = db.query(Question).filter(Question.corpus_id == corpus_id).count()
    return {"corpus_id": corpus_id, "question_count": count}


@router.get("/corpus/{corpus_id}/types")
async def get_question_types(corpus_id: int, db: Session = Depends(get_db)):
    """Get statistics about question types for a corpus"""
    questions = db.query(Question).filter(Question.corpus_id == corpus_id).all()
    
    # Count by generation method
    manual_count = sum(1 for q in questions if q.generated_by == "manual")
    ai_count = sum(1 for q in questions if q.generated_by == "ai")
    
    return {
        "corpus_id": corpus_id,
        "total_questions": len(questions),
        "manual_questions": manual_count,
        "ai_generated_questions": ai_count
    } 