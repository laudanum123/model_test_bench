from typing import List, Dict, Any, Optional
import re
from app.services.llm_service import LLMServiceFactory, LLMProvider
from app.models.schemas import QuestionCreate
from app.database import Question, get_db
from sqlalchemy.orm import Session


class QuestionGeneratorService:
    """Service for generating synthetic questions from text corpora"""
    
    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI, 
                 llm_model: str = "gpt-3.5-turbo"):
        self.llm_service = LLMServiceFactory.create_service(llm_provider, llm_model)
    
    async def generate_questions_from_text(self, text: str, num_questions: int = 5, 
                                         question_types: List[str] = None) -> List[Dict[str, str]]:
        """Generate questions from a given text"""
        if question_types is None:
            question_types = ["factual", "inferential", "analytical"]
        
        # Split text into chunks if it's too long
        chunks = self._split_text_into_chunks(text, max_chunk_size=2000)
        
        questions = []
        questions_per_chunk = max(1, num_questions // len(chunks))
        
        for chunk in chunks:
            chunk_questions = await self._generate_questions_for_chunk(
                chunk, questions_per_chunk, question_types
            )
            questions.extend(chunk_questions)
        
        # Limit to requested number of questions
        return questions[:num_questions]
    
    async def _generate_questions_for_chunk(self, chunk: str, num_questions: int, 
                                          question_types: List[str]) -> List[Dict[str, str]]:
        """Generate questions for a specific text chunk"""
        try:
            prompt = f"""Generate {num_questions} diverse questions and their reference answers based on the following text. 
            Include different types of questions: {', '.join(question_types)}.

            Text: {chunk}

            For each question, provide:
            1. A clear, specific question
            2. A comprehensive reference answer based on the text
            3. The question type (factual, inferential, analytical, etc.)

            Format your response as:
            Q1: [Question]
            A1: [Reference Answer]
            T1: [Question Type]

            Q2: [Question]
            A2: [Reference Answer]
            T2: [Question Type]

            ... and so on.

            Make sure questions are:
            - Specific and answerable from the text
            - Varied in difficulty and type
            - Clear and unambiguous
            - Cover different aspects of the content"""

            response = await self.llm_service.generate(prompt, temperature=0.7)
            
            return self._parse_qa_pairs(response)
        
        except Exception as e:
            print(f"Error generating questions for chunk: {str(e)}")
            return []
    
    def _parse_qa_pairs(self, response: str) -> List[Dict[str, str]]:
        """Parse the LLM response to extract Q&A pairs"""
        qa_pairs = []
        
        # Split by question patterns
        question_pattern = r'Q\d+:\s*(.+?)(?=\nA\d+:|$)'
        answer_pattern = r'A\d+:\s*(.+?)(?=\nQ\d+:|$)'
        type_pattern = r'T\d+:\s*(.+?)(?=\nQ\d+:|$)'
        
        questions = re.findall(question_pattern, response, re.DOTALL)
        answers = re.findall(answer_pattern, response, re.DOTALL)
        types = re.findall(type_pattern, response, re.DOTALL)
        
        # Match questions with answers and types
        for i in range(min(len(questions), len(answers))):
            qa_pair = {
                'question': questions[i].strip(),
                'answer': answers[i].strip(),
                'type': types[i].strip() if i < len(types) else 'factual'
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """Split text into manageable chunks"""
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def save_questions_to_database(self, db: Session, corpus_id: int, 
                                       questions: List[Dict[str, str]]) -> List[Question]:
        """Save generated questions to the database"""
        saved_questions = []
        
        for qa_pair in questions:
            try:
                question = Question(
                    corpus_id=corpus_id,
                    question_text=qa_pair['question'],
                    reference_answer=qa_pair['answer'],
                    generated_by="ai"
                )
                
                db.add(question)
                db.commit()
                db.refresh(question)
                saved_questions.append(question)
            
            except Exception as e:
                db.rollback()
                print(f"Error saving question: {str(e)}")
                continue
        
        return saved_questions
    
    async def generate_questions_from_corpus(self, db: Session, corpus_id: int, 
                                           corpus_text: str, num_questions: int = 5) -> List[Question]:
        """Generate and save questions for a corpus"""
        try:
            # Generate questions
            qa_pairs = await self.generate_questions_from_text(corpus_text, num_questions)
            
            # Save to database
            saved_questions = await self.save_questions_to_database(db, corpus_id, qa_pairs)
            
            return saved_questions
        
        except Exception as e:
            raise Exception(f"Error generating questions from corpus: {str(e)}")
    
    async def generate_questions_by_topic(self, text: str, topics: List[str], 
                                        questions_per_topic: int = 3) -> List[Dict[str, str]]:
        """Generate questions focused on specific topics"""
        all_questions = []
        
        for topic in topics:
            try:
                prompt = f"""Generate {questions_per_topic} questions specifically about "{topic}" based on the following text.

                Text: {text}
                Topic: {topic}

                Focus on questions that:
                - Are specifically about {topic}
                - Can be answered using information from the text
                - Cover different aspects of {topic}
                - Are clear and specific

                Format your response as:
                Q1: [Question about {topic}]
                A1: [Reference Answer]
                T1: [Question Type]

                Q2: [Question about {topic}]
                A2: [Reference Answer]
                T2: [Question Type]

                ... and so on."""

                response = await self.llm_service.generate(prompt, temperature=0.7)
                topic_questions = self._parse_qa_pairs(response)
                all_questions.extend(topic_questions)
            
            except Exception as e:
                print(f"Error generating questions for topic {topic}: {str(e)}")
                continue
        
        return all_questions 