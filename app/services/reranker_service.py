from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
import numpy as np


class RerankerService(ABC):
    """Abstract base class for reranker services"""
    
    @abstractmethod
    async def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """Rerank documents based on relevance to query"""
        pass


class TransformersRerankerService(RerankerService):
    """Transformers-based reranker service"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the reranker model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
        except Exception as e:
            raise Exception(f"Failed to load reranker model {self.model_name}: {str(e)}")
    
    async def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        try:
            if not documents:
                return []
            
            # Prepare pairs for scoring
            pairs = []
            for doc in documents:
                # Format depends on the model - some expect [query, document] format
                if "bge" in self.model_name.lower():
                    pair = f"{query} [SEP] {doc}"
                else:
                    pair = f"{query} {self.tokenizer.sep_token} {doc}"
                pairs.append(pair)
            
            # Tokenize and get scores
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                # Get relevance scores (usually the positive class)
                relevance_scores = scores[:, 1].cpu().numpy()
            
            # Create (index, score) pairs and sort by score
            scored_docs = list(enumerate(relevance_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            if top_k is not None:
                scored_docs = scored_docs[:top_k]
            
            return scored_docs
        except Exception as e:
            raise Exception(f"Reranker error: {str(e)}")


class CrossEncoderRerankerService(RerankerService):
    """Sentence Transformers CrossEncoder reranker service"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            raise Exception(f"Failed to load cross-encoder model {self.model_name}: {str(e)}")
    
    async def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        try:
            if not documents:
                return []
            
            # Prepare pairs for scoring
            pairs = [[query, doc] for doc in documents]
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Create (index, score) pairs and sort by score
            scored_docs = list(enumerate(scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            if top_k is not None:
                scored_docs = scored_docs[:top_k]
            
            return scored_docs
        except Exception as e:
            raise Exception(f"Cross-encoder reranker error: {str(e)}")


class RerankerServiceFactory:
    """Factory for creating reranker services"""
    
    @staticmethod
    def create_service(model_name: str) -> RerankerService:
        # Determine the type of model based on the name
        if "cross-encoder" in model_name.lower():
            return CrossEncoderRerankerService(model_name)
        else:
            return TransformersRerankerService(model_name)