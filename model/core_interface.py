from abc import ABC, abstractmethod
from typing import List, Any
# Using LangChain's Document object as a standard for passing data
from langchain_core.documents import Document

class EmbeddingInterface(ABC):
    """Interface for an embedding model."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query text."""
        pass

class LLMInterface(ABC):
    """Interface for a Language Model."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: List[Document]) -> str:
        """Generates a response based on a prompt and context."""
        pass

class VectorStoreInterface(ABC):
    """Interface for a vector database."""
    
    @abstractmethod
    def upsert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Upserts documents and their embeddings into the store."""
        pass
        
    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """Queries the store for similar documents."""
        pass

class DataLoaderInterface(ABC):
    """Interface for loading data."""
    
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """Loads data from a source (e.g., directory, file)."""
        pass

class TextSplitterInterface(ABC):
    """Interface for splitting text."""
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents into smaller chunks."""
        pass
