from typing import List, Dict, Any, Optional
import asyncio
from app.services.llm_service import LLMServiceFactory, LLMProvider
from app.services.embedding_service import EmbeddingServiceFactory, EmbeddingProvider
from app.models.schemas import EvaluationResultCreate
from app.database import EvaluationResult, get_db
from sqlalchemy.orm import Session


class EvaluationService:
    """Service for evaluating LLM responses and retrieval quality"""
    
    def __init__(self, judge_provider: LLMProvider = LLMProvider.OPENAI, 
                 judge_model: str = "gpt-3.5-turbo"):
        self.judge_service = LLMServiceFactory.create_service(judge_provider, judge_model)
        self.embedding_service = EmbeddingServiceFactory.create_service(
            EmbeddingProvider.SENTENCE_TRANSFORMERS, "all-MiniLM-L6-v2"
        )
    
    async def evaluate_answer_relevance(self, question: str, reference_answer: str, 
                                      generated_answer: str) -> float:
        """Evaluate how well the generated answer matches the reference answer"""
        try:
            prompt = f"""You are an expert evaluator. Rate how well the generated answer matches the reference answer for the given question.

Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Rate the relevance on a scale of 0.0 to 1.0, where:
- 0.0: Completely irrelevant or incorrect
- 0.5: Partially relevant but missing key information
- 1.0: Highly relevant and accurate

Consider:
1. Factual accuracy
2. Completeness of information
3. Relevance to the question
4. Clarity and coherence

Provide only the numerical score (0.0-1.0):"""

            response = await self.judge_service.generate(prompt, temperature=0.1)
            
            # Extract numerical score
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                # Fallback to embedding similarity
                return await self._calculate_similarity(reference_answer, generated_answer)
        
        except Exception as e:
            print(f"Error in answer relevance evaluation: {str(e)}")
            # Fallback to embedding similarity
            return await self._calculate_similarity(reference_answer, generated_answer)
    
    async def evaluate_retrieval_relevance(self, question: str, retrieved_chunks: List[str]) -> float:
        """Evaluate how relevant the retrieved chunks are to the question"""
        try:
            if not retrieved_chunks:
                return 0.0
            
            # Combine all chunks
            combined_chunks = "\n\n".join(retrieved_chunks)
            
            prompt = f"""You are an expert evaluator. Rate how relevant the retrieved information is to the given question.

Question: {question}
Retrieved Information: {combined_chunks}

Rate the relevance on a scale of 0.0 to 1.0, where:
- 0.0: Completely irrelevant information
- 0.5: Somewhat relevant but not very useful
- 1.0: Highly relevant and useful for answering the question

Consider:
1. How well the information addresses the question
2. Whether key concepts from the question are present
3. The usefulness of the information for generating an answer

Provide only the numerical score (0.0-1.0):"""

            response = await self.judge_service.generate(prompt, temperature=0.1)
            
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                # Fallback to embedding similarity
                return await self._calculate_similarity(question, combined_chunks)
        
        except Exception as e:
            print(f"Error in retrieval relevance evaluation: {str(e)}")
            # Fallback to embedding similarity
            return await self._calculate_similarity(question, combined_chunks)
    
    async def evaluate_chunk_overlap(self, question: str, reference_answer: str, 
                                   retrieved_chunks: List[str]) -> float:
        """Evaluate how much relevant information from the reference answer is present in retrieved chunks"""
        try:
            if not retrieved_chunks:
                return 0.0
            
            combined_chunks = "\n\n".join(retrieved_chunks)
            
            prompt = f"""You are an expert evaluator. Rate how much of the information needed to answer the question correctly is present in the retrieved chunks.

Question: {question}
Reference Answer: {reference_answer}
Retrieved Chunks: {combined_chunks}

Rate the information coverage on a scale of 0.0 to 1.0, where:
- 0.0: No relevant information from the reference answer is present
- 0.5: Some relevant information is present but incomplete
- 1.0: All or most relevant information from the reference answer is present

Consider:
1. Key facts and details from the reference answer
2. Important concepts and relationships
3. Specific information needed to answer the question

Provide only the numerical score (0.0-1.0):"""

            response = await self.judge_service.generate(prompt, temperature=0.1)
            
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                # Fallback to embedding similarity
                return await self._calculate_similarity(reference_answer, combined_chunks)
        
        except Exception as e:
            print(f"Error in chunk overlap evaluation: {str(e)}")
            # Fallback to embedding similarity
            return await self._calculate_similarity(reference_answer, combined_chunks)
    
    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings"""
        try:
            similarity = await self.embedding_service.similarity(text1, text2)
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    async def evaluate_single_result(self, question: str, reference_answer: str, 
                                   generated_answer: str, retrieved_chunks: List[str]) -> Dict[str, float]:
        """Evaluate a single question-answer pair"""
        # Run all evaluations concurrently
        tasks = [
            self.evaluate_answer_relevance(question, reference_answer, generated_answer),
            self.evaluate_retrieval_relevance(question, retrieved_chunks),
            self.evaluate_chunk_overlap(question, reference_answer, retrieved_chunks)
        ]
        
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        answer_relevance = scores[0] if not isinstance(scores[0], Exception) else 0.0
        retrieval_relevance = scores[1] if not isinstance(scores[1], Exception) else 0.0
        chunk_overlap = scores[2] if not isinstance(scores[2], Exception) else 0.0
        
        return {
            'answer_relevance_score': answer_relevance,
            'retrieval_relevance_score': retrieval_relevance,
            'chunk_overlap_score': chunk_overlap
        }
    
    async def save_evaluation_result(self, db: Session, evaluation_run_id: int, 
                                   question_id: int, generated_answer: str, 
                                   retrieved_chunks: List[Dict[str, Any]], 
                                   scores: Dict[str, float]) -> EvaluationResult:
        """Save evaluation result to database"""
        try:
            evaluation_result = EvaluationResult(
                evaluation_run_id=evaluation_run_id,
                question_id=question_id,
                generated_answer=generated_answer,
                retrieved_chunks=retrieved_chunks,
                answer_relevance_score=scores['answer_relevance_score'],
                retrieval_relevance_score=scores['retrieval_relevance_score'],
                chunk_overlap_score=scores['chunk_overlap_score'],
                evaluation_details=scores
            )
            
            db.add(evaluation_result)
            db.commit()
            db.refresh(evaluation_result)
            
            return evaluation_result
        except Exception as e:
            db.rollback()
            raise Exception(f"Error saving evaluation result: {str(e)}") 