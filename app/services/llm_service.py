from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from openai import AsyncOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.config import settings
from app.models.schemas import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class LLMService(ABC):
    """Abstract base class for LLM services"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    async def generate_with_context(self, question: str, context: List[str], **kwargs) -> str:
        """Generate answer with context"""
        pass


class OpenAIService(LLMService):
    """OpenAI API service"""
    
    def __init__(self, model: str = "gpt-4.1"):
        logger.info(f"Initializing OpenAI service with model: {model}")
        self.model = model
        if settings.openai_api_key:
            logger.debug("OpenAI API key found in settings")
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            logger.error("OpenAI API key not configured")
            raise ValueError("OpenAI API key not configured")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        logger.debug(f"Generating with OpenAI model: {self.model}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        logger.debug(f"Additional kwargs: {kwargs}")
        
        try:
            logger.debug("Sending request to OpenAI API")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            result = response.choices[0].message.content
            logger.debug(f"Received response from OpenAI, length: {len(result)} characters")
            return result
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def generate_with_context(self, question: str, context: List[str], **kwargs) -> str:
        logger.debug(f"Generating with context, question length: {len(question)}, context items: {len(context)}")
        context_text = "\n\n".join(context)
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""
        
        return await self.generate(prompt, **kwargs)


class TransformersService(LLMService):
    """Local Transformers service"""
    
    def __init__(self, model_name: str):
        logger.info(f"Initializing Transformers service with model: {model_name}")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        logger.debug(f"Loading model and tokenizer for: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.debug("Tokenizer loaded successfully")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.debug("Model loaded successfully")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.debug("Set pad_token to eos_token")
                
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to load model {self.model_name}: {str(e)}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        logger.debug(f"Generating with Transformers model: {self.model_name}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        logger.debug(f"Additional kwargs: {kwargs}")
        
        try:
            logger.debug("Tokenizing input")
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            logger.debug(f"Input tokens: {inputs.input_ids.shape}")
            
            logger.debug("Generating response")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=kwargs.get("max_new_tokens", 512),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=kwargs.get("do_sample", True),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            logger.debug("Decoding response")
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            response = response[len(prompt):].strip()
            logger.debug(f"Generated response length: {len(response)} characters")
            return response
        except Exception as e:
            logger.error(f"Transformers generation error: {str(e)}", exc_info=True)
            raise Exception(f"Transformers generation error: {str(e)}")
    
    async def generate_with_context(self, question: str, context: List[str], **kwargs) -> str:
        logger.debug(f"Generating with context, question length: {len(question)}, context items: {len(context)}")
        context_text = "\n\n".join(context)
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""
        
        return await self.generate(prompt, **kwargs)


class LLMServiceFactory:
    """Factory for creating LLM services"""
    
    @staticmethod
    def create_service(provider: LLMProvider, model: str) -> LLMService:
        logger.info(f"Creating LLM service with provider: {provider}, model: {model}")
        try:
            if provider == LLMProvider.OPENAI:
                logger.debug("Creating OpenAI service")
                return OpenAIService(model)
            elif provider == LLMProvider.TRANSFORMERS:
                logger.debug("Creating Transformers service")
                return TransformersService(model)
            else:
                logger.error(f"Unsupported LLM provider: {provider}")
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to create LLM service: {str(e)}", exc_info=True)
            raise