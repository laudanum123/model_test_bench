from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from rank_bm25 import BM25Okapi

from app.models.schemas import RetrievalStrategy, VectorStore
from app.services.embedding_service import EmbeddingProvider, EmbeddingServiceFactory
from app.services.reranker_service import RerankerServiceFactory
from app.services.vector_store_service import VectorStoreServiceFactory
from app.typing_helpers.vector_store_protocols import SupportsUpdateEmbeddings


class RetrievalService(ABC):
    """Abstract base class for retrieval services"""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        pass

    @abstractmethod
    async def add_documents(self, documents: list[str], metadata: list[dict[str, Any]] | None = None):
        """Add documents to the retrieval system"""
        pass


class SemanticRetrievalService(RetrievalService):
    """Semantic retrieval using embeddings and vector search"""

    def __init__(self, embedding_provider: EmbeddingProvider, embedding_model: str,
                 vector_store: VectorStore, reranker_model: str | None = None):
        self.embedding_service = EmbeddingServiceFactory.create_service(embedding_provider, embedding_model)
        self.vector_store = VectorStoreServiceFactory.create_service(vector_store.value)
        self.reranker = None
        if reranker_model:
            self.reranker = RerankerServiceFactory.create_service(reranker_model)

    async def add_documents(self, documents: list[str], metadata: list[dict[str, Any]] | None = None):
        """Add documents to the vector store"""
        # Add to vector store first
        await self.vector_store.add_documents(documents, metadata)

        # Only generate and update embeddings for FAISS (ChromaDB handles embeddings internally)
        if isinstance(self.vector_store, SupportsUpdateEmbeddings):
            # Generate embeddings for documents
            embeddings = await self.embedding_service.embed(documents)

            # Ensure embeddings is List[List[float]] for multiple documents
            if isinstance(embeddings[0], list):
                await self.vector_store.update_embeddings(cast("list[list[float]]", embeddings))
            else:
                await self.vector_store.update_embeddings(cast("list[list[float]]", [embeddings]))

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve documents using semantic search"""
        # Generate query embedding
        query_embedding = await self.embedding_service.embed(query)

        # Ensure query_embedding is List[float] for single query
        if isinstance(query_embedding[0], list):
            query_embedding = cast("list[float]", query_embedding[0])

        # Search vector store
        results = await self.vector_store.search(cast("list[float]", query_embedding), top_k)

        # Apply reranker if available
        if self.reranker and results:
            documents = [result["text"] for result in results]
            reranked_indices = await self.reranker.rerank(query, documents, top_k)

            # Reorder results based on reranker
            reranked_results = []
            for idx, score in reranked_indices:
                if idx < len(results):
                    result = results[idx].copy()
                    result["reranker_score"] = score
                    reranked_results.append(result)

            return reranked_results

        return results


class BM25RetrievalService(RetrievalService):
    """BM25 retrieval using traditional keyword search"""

    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.metadata = []

    async def add_documents(self, documents: list[str], metadata: list[dict[str, Any]] | None = None):
        """Add documents to BM25 index"""
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"source": "upload"} for _ in documents])

        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve documents using BM25"""
        if not self.bm25 or not self.documents:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(scores[idx]),
                    "id": f"doc_{idx}"
                })

        return results


class HybridRetrievalService(RetrievalService):
    """Hybrid retrieval combining semantic and BM25 search"""

    def __init__(self, embedding_provider: EmbeddingProvider, embedding_model: str,
                 vector_store: VectorStore, reranker_model: str | None = None,
                 semantic_weight: float = 0.7):
        self.semantic_service = SemanticRetrievalService(
            embedding_provider, embedding_model, vector_store, reranker_model
        )
        self.bm25_service = BM25RetrievalService()
        self.semantic_weight = semantic_weight
        self.bm25_weight = 1.0 - semantic_weight

    async def add_documents(self, documents: list[str], metadata: list[dict[str, Any]] | None = None):
        """Add documents to both retrieval systems"""
        await self.semantic_service.add_documents(documents, metadata)
        await self.bm25_service.add_documents(documents, metadata)

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve documents using hybrid search"""
        # Get results from both systems
        semantic_results = await self.semantic_service.retrieve(query, top_k * 2)
        bm25_results = await self.bm25_service.retrieve(query, top_k * 2)

        # Combine and score results
        combined_results = {}

        # Process semantic results
        for result in semantic_results:
            doc_id = result["id"]
            semantic_score = result.get("distance", 0) or result.get("reranker_score", 0)
            combined_results[doc_id] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "semantic_score": semantic_score,
                "bm25_score": 0,
                "combined_score": semantic_score * self.semantic_weight
            }

        # Process BM25 results
        for result in bm25_results:
            doc_id = result["id"]
            bm25_score = result.get("score", 0)

            if doc_id in combined_results:
                combined_results[doc_id]["bm25_score"] = bm25_score
                combined_results[doc_id]["combined_score"] += bm25_score * self.bm25_weight
            else:
                combined_results[doc_id] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_score": 0,
                    "bm25_score": bm25_score,
                    "combined_score": bm25_score * self.bm25_weight
                }

        # Sort by combined score and return top-k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]

        # Format results
        formatted_results = []
        for result in sorted_results:
            formatted_results.append({
                "text": result["text"],
                "metadata": result["metadata"],
                "combined_score": result["combined_score"],
                "semantic_score": result["semantic_score"],
                "bm25_score": result["bm25_score"],
                "id": result.get("id", "unknown")
            })

        return formatted_results


class RetrievalServiceFactory:
    """Factory for creating retrieval services"""

    @staticmethod
    def create_service(strategy: RetrievalStrategy, **kwargs) -> RetrievalService:
        if strategy == RetrievalStrategy.SEMANTIC:
            return SemanticRetrievalService(**kwargs)
        elif strategy == RetrievalStrategy.BM25:
            return BM25RetrievalService(**kwargs)
        elif strategy == RetrievalStrategy.HYBRID:
            return HybridRetrievalService(**kwargs)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")
