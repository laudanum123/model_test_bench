from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Metadata
import faiss
import numpy as np
from app.config import settings


class VectorStoreService(ABC):
    """Abstract base class for vector store services"""
    
    @abstractmethod
    async def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        pass

    
    @abstractmethod
    async def clear(self):
        """Clear all documents from the vector store"""
        pass


class ChromaService(VectorStoreService):
    """ChromaDB vector store service"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="./data/chroma")
        self.collection = None
        self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create the collection"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    async def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        try:
            if metadata is None:
                metadata = [{"source": "upload"} for _ in documents]
            
            # Convert metadata to ChromaDB Metadata type
            chroma_metadata: List[Metadata] = []
            for meta in metadata:
                chroma_meta: Metadata = {}
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        chroma_meta[key] = value
                chroma_metadata.append(chroma_meta)
            
            # Generate IDs for documents
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            if self.collection is None:
                raise Exception("Failed to initialize ChromaDB collection")
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=chroma_metadata,
                ids=ids
            )
            
            return ids
        except Exception as e:
            raise Exception(f"ChromaDB add documents error: {str(e)}")
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""	
        try:
            # Convert embedding to list format expected by ChromaDB
            query_embedding_array = np.array([query_embedding], dtype=np.float32)
            
            if self.collection is None:
                raise Exception("Failed to initialize ChromaDB collection")
            
            results = self.collection.query(
                query_embeddings=query_embedding_array,
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results and 'documents' and 'metadatas' and 'distances' and 'ids' in results and results['documents'] and results['metadatas'] and results['distances'] and results['ids']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i], 
                        'distance': results['distances'][0][i],
                        'id': results['ids'][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            raise Exception(f"ChromaDB search error: {str(e)}")
    
    async def clear(self):
        try:
            self.client.delete_collection(self.collection_name)
            self._get_or_create_collection()
        except Exception as e:
            raise Exception(f"ChromaDB clear error: {str(e)}")


class FAISSService(VectorStoreService):
    """FAISS vector store service"""
    
    def __init__(self, dimension: int = 384, index_type: str = "cosine"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.metadata = []
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index"""
        if self.index_type == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        elif self.index_type == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    async def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        try:
            if metadata is None:
                metadata = [{"source": "upload"} for _ in documents]
            
            # Store documents and metadata
            self.documents.extend(documents)
            self.metadata.extend(metadata)
            
            # Generate IDs
            ids = [f"doc_{len(self.documents) - len(documents) + i}" for i in range(len(documents))]
            
            return ids
        except Exception as e:
            raise Exception(f"FAISS add documents error: {str(e)}")
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.documents or self.index is None:
                return []
            
            # Convert query embedding to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize query vector for cosine similarity
            if isinstance(self.index, faiss.IndexFlatIP):
                faiss.normalize_L2(query_vector)

            # Search
            k = min(top_k, len(self.documents))
            if self.index is not None:
            
                distances, indices = self.index.search(x=query_vector, k=k) # type: ignore[call-arg]
            else:
                return []
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'distance': float(distance),
                        'id': f"doc_{idx}"
                    })
            
            return results
        except Exception as e:
            raise Exception(f"FAISS search error: {str(e)}")
    
    async def clear(self):
        try:
            self.documents = []
            self.metadata = []
            self._create_index()
        except Exception as e:
            raise Exception(f"FAISS clear error: {str(e)}")
    
    async def update_embeddings(self, embeddings: List[List[float]]):
        """Update the FAISS index with new embeddings"""
        try:
            if embeddings and self.index is not None:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Normalize embeddings for cosine similarity
                if isinstance(self.index, faiss.IndexFlatIP):
                    faiss.normalize_L2(embeddings_array)
                
                n = embeddings_array.shape[0]
                self.index.add(x=embeddings_array, n=n)
        except Exception as e:
            raise Exception(f"FAISS update embeddings error: {str(e)}")


class VectorStoreServiceFactory:
    """Factory for creating vector store services"""
    
    @staticmethod
    def create_service(store_type: str, **kwargs) -> VectorStoreService:
        if store_type == "chroma":
            return ChromaService(**kwargs)
        elif store_type == "faiss":
            return FAISSService(**kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}") 