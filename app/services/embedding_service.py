from abc import ABC, abstractmethod
from typing import List, Union, cast
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.models.schemas import EmbeddingProvider


class EmbeddingService(ABC):
    """Abstract base class for embedding services"""
    
    @abstractmethod
    async def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text(s)"""
        pass
    
    @abstractmethod
    async def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        pass


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service"""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
        else:
            raise ValueError("OpenAI API key not configured")
    
    async def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            response = openai.embeddings.create(
                input=texts,
                model=self.model
            )
            
            embeddings = [data.embedding for data in response.data]
            return embeddings[0] if len(embeddings) == 1 else embeddings
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")
    
    async def similarity(self, text1: str, text2: str) -> float:
        embeddings = await self.embed([text1, text2])
        if isinstance(embeddings[0], list):
            emb1, emb2 = embeddings
        else:
            emb1, emb2 = embeddings[0], embeddings[1]
        
        return self._cosine_similarity(cast(List[float], emb1), cast(List[float], emb2))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        return np.dot(vec1_array, vec2_array) / (np.linalg.norm(vec1_array) * np.linalg.norm(vec2_array))


class SentenceTransformersService(EmbeddingService):
    """Sentence Transformers embedding service"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise Exception(f"Failed to load model {self.model_name}: {str(e)}")
    
    async def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            if self.model:
                embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            if len(texts) == 1:
                return embeddings[0].tolist()
            else:
                return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Sentence Transformers embedding error: {str(e)}")
    
    async def similarity(self, text1: str, text2: str) -> float:
        embeddings = await self.embed([text1, text2])
        if isinstance(embeddings[0], list):
            emb1, emb2 = embeddings
        else:
            emb1, emb2 = embeddings[0], embeddings[1]
        
        return self._cosine_similarity(cast(List[float], emb1), cast(List[float], emb2))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        return np.dot(vec1_array, vec2_array) / (np.linalg.norm(vec1_array) * np.linalg.norm(vec2_array))


class EmbeddingServiceFactory:
    """Factory for creating embedding services"""
    
    @staticmethod
    def create_service(provider: EmbeddingProvider, model: str) -> EmbeddingService:
        if provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddingService(model)
        elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformersService(model)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")