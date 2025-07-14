import logging
import random
import re

from sqlalchemy.orm import Session

from app.database import Question
from app.services.llm_service import LLMProvider, LLMServiceFactory

# Configure logging
logger = logging.getLogger(__name__)


class QuestionGeneratorService:
    """Service for generating synthetic questions from text corpora"""

    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI,
                 llm_model: str = "gpt-4.1"):
        logger.info(f"Initializing QuestionGeneratorService with provider={llm_provider}, model={llm_model}")
        try:
            self.llm_service = LLMServiceFactory.create_service(llm_provider, llm_model)
            logger.info("LLM service created successfully")
        except Exception as e:
            logger.error(f"Failed to create LLM service: {e!s}")
            raise

    async def generate_questions_from_text(self, text: str, num_questions: int = 5,
                                         question_types: list[str] | None = None) -> list[dict[str, str]]:
        """Generate questions from a given text"""
        logger.info(f"Starting question generation from text, num_questions={num_questions}")
        logger.debug(f"Input text length: {len(text)} characters")

        if question_types is None:
            question_types = ["factual", "inferential", "analytical"]

        logger.debug(f"Question types: {question_types}")

        # Split text into chunks if it's too long
        logger.debug("Splitting text into chunks")
        chunks = self._split_text_into_chunks(text, max_chunk_size=2000)
        logger.info(f"Split text into {len(chunks)} chunks")

        questions = []

        # Determine how many chunks to select and questions per chunk
        if num_questions <= len(chunks):
            # Select random chunks without replacement
            selected_chunks = random.sample(chunks, num_questions)
            questions_per_chunk = 1
            logger.debug(f"Selected {len(selected_chunks)} chunks (1 question per chunk)")
        else:
            # If more questions than chunks, use all chunks and distribute questions
            selected_chunks = chunks
            questions_per_chunk = max(1, num_questions // len(chunks))
            logger.debug(f"Using all {len(selected_chunks)} chunks with {questions_per_chunk} questions per chunk")

        for i, chunk in enumerate(selected_chunks):
            logger.debug(f"Processing chunk {i+1}/{len(selected_chunks)}, length: {len(chunk)} characters")
            chunk_questions = await self._generate_questions_for_chunk(
                chunk, questions_per_chunk, question_types
            )
            logger.debug(f"Generated {len(chunk_questions)} questions for chunk {i+1}")
            questions.extend(chunk_questions)

        # Limit to requested number of questions
        final_questions = questions[:num_questions]
        logger.info(f"Final result: {len(final_questions)} questions generated")
        return final_questions

    async def _generate_questions_for_chunk(self, chunk: str, num_questions: int,
                                          question_types: list[str]) -> list[dict[str, str]]:
        """Generate questions for a specific text chunk"""
        logger.debug(f"Generating {num_questions} questions for chunk")
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

            logger.debug("Sending prompt to LLM service")
            response = await self.llm_service.generate(prompt, temperature=0.7)
            logger.debug(f"Received response from LLM, length: {len(response)} characters")

            qa_pairs = self._parse_qa_pairs(response, num_questions)
            logger.debug(f"Parsed {len(qa_pairs)} Q&A pairs from response")
            return qa_pairs

        except Exception as e:
            logger.error(f"Error generating questions for chunk: {e!s}", exc_info=True)
            return []

    def _parse_qa_pairs(self, response: str, num_questions: int | None = None) -> list[dict[str, str]]:
        """Parse the LLM response to extract Q&A pairs"""
        logger.debug("Parsing Q&A pairs from LLM response")
        qa_pairs = []

        # Split by question patterns
        question_pattern = r"Q\d+:\s*(.+?)(?=\nA\d+:|$)"
        answer_pattern = r"A\d+:\s*(.+?)(?=\nQ\d+:|$)"
        type_pattern = r"T\d+:\s*(.+?)(?=\nQ\d+:|$)"

        questions = re.findall(question_pattern, response, re.DOTALL)
        answers = re.findall(answer_pattern, response, re.DOTALL)
        types = re.findall(type_pattern, response, re.DOTALL)

        logger.debug(f"Found {len(questions)} questions, {len(answers)} answers, {len(types)} types")

        # Match questions with answers and types
        for i in range(min(len(questions), len(answers))):
            qa_pair = {
                "question": questions[i].strip(),
                "answer": answers[i].strip(),
                "type": types[i].strip() if i < len(types) else "factual"
            }
            qa_pairs.append(qa_pair)
            logger.debug(f"Parsed Q&A pair {i+1}: {qa_pair['question'][:50]}...")

        # Limit to requested number of questions if specified
        if num_questions is not None and len(qa_pairs) > num_questions:
            logger.info(f"Limiting parsed questions from {len(qa_pairs)} to {num_questions}")
            qa_pairs = qa_pairs[:num_questions]

        logger.info(f"Successfully parsed {len(qa_pairs)} Q&A pairs")
        return qa_pairs

    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 2000) -> list[str]:
        """Split text into manageable chunks"""
        logger.debug(f"Splitting text into chunks, max_chunk_size={max_chunk_size}")

        if len(text) <= max_chunk_size:
            logger.debug("Text is small enough, returning as single chunk")
            return [text]

        # Split by sentences first
        sentences = re.split(r"[.!?]+", text)
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

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    async def save_questions_to_database(self, db: Session, corpus_id: int,
                                       questions: list[dict[str, str]]) -> list[Question]:
        """Save generated questions to the database"""
        logger.info(f"Saving {len(questions)} questions to database for corpus_id: {corpus_id}")
        saved_questions = []

        for i, qa_pair in enumerate(questions):
            try:
                logger.debug(f"Saving question {i+1}/{len(questions)}")
                question = Question(
                    corpus_id=corpus_id,
                    question_text=qa_pair["question"],
                    reference_answer=qa_pair["answer"],
                    generated_by="ai"
                )

                db.add(question)
                db.commit()
                db.refresh(question)
                saved_questions.append(question)
                logger.debug(f"Successfully saved question with id: {question.id}")

            except Exception as e:
                logger.error(f"Error saving question {i+1}: {e!s}")
                db.rollback()
                continue

        logger.info(f"Successfully saved {len(saved_questions)} questions to database")
        return saved_questions

    async def generate_questions_from_corpus(self, db: Session, corpus_id: int,
                                           corpus_text: str, num_questions: int = 5) -> list[Question]:
        """Generate and save questions for a corpus"""
        logger.info(f"Starting question generation from corpus {corpus_id}, num_questions={num_questions}")
        try:
            # Generate questions
            logger.debug("Generating questions from text")
            qa_pairs = await self.generate_questions_from_text(corpus_text, num_questions)
            logger.info(f"Generated {len(qa_pairs)} Q&A pairs")

            # Save to database
            logger.debug("Saving questions to database")
            saved_questions = await self.save_questions_to_database(db, corpus_id, qa_pairs)
            logger.info(f"Successfully saved {len(saved_questions)} questions to database")

            return saved_questions

        except Exception as e:
            logger.error(f"Error generating questions from corpus: {e!s}", exc_info=True)
            raise Exception(f"Error generating questions from corpus: {e!s}") from e

    async def generate_questions_by_topic(self, text: str, topics: list[str],
                                        questions_per_topic: int = 3) -> list[dict[str, str]]:
        """Generate questions focused on specific topics"""
        logger.info(f"Starting topic-based question generation for {len(topics)} topics")
        logger.debug(f"Topics: {topics}, questions_per_topic: {questions_per_topic}")

        all_questions = []

        for i, topic in enumerate(topics):
            logger.info(f"Generating questions for topic {i+1}/{len(topics)}: {topic}")
            try:
                prompt = f"""Generate {questions_per_topic} questions specifically about "{topic}" based on the following text.

                Text: {text}
                Topic: {topic}

                Focus on questions that:
                - Resemble natural user queries to a general knowledge chatbot
                - Explicitly mention the topic in the question (don't assume the chatbot knows the context)
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

                logger.debug(f"Sending topic-specific prompt to LLM for topic: {topic}")
                response = await self.llm_service.generate(prompt, temperature=0.7)
                logger.debug(f"Received response for topic '{topic}', length: {len(response)} characters")

                topic_questions = self._parse_qa_pairs(response, questions_per_topic)
                logger.info(f"Generated {len(topic_questions)} questions for topic '{topic}'")
                all_questions.extend(topic_questions)

            except Exception as e:
                logger.error(f"Error generating questions for topic '{topic}': {e!s}", exc_info=True)
                continue

        logger.info(f"Total questions generated across all topics: {len(all_questions)}")
        return all_questions
