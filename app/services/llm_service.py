from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.config import settings
from app.models.schemas import LLMProvider


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
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
        else:
            raise ValueError("OpenAI API key not configured")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def generate_with_context(self, question: str, context: List[str], **kwargs) -> str:
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
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise Exception(f"Failed to load model {self.model_name}: {str(e)}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=kwargs.get("max_new_tokens", 512),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=kwargs.get("do_sample", True),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            response = response[len(prompt):].strip()
            return response
        except Exception as e:
            raise Exception(f"Transformers generation error: {str(e)}")
    
    async def generate_with_context(self, question: str, context: List[str], **kwargs) -> str:
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
        if provider == LLMProvider.OPENAI:
            return OpenAIService(model)
        elif provider == LLMProvider.TRANSFORMERS:
            return TransformersService(model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")