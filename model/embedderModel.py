import os
from typing import List
from langchain_community.embeddings import (
    HuggingFaceEmbeddings, 
    GoogleGenerativeAiEmbeddings
)
from core.interfaces import EmbeddingInterface
import config

class SentenceTransformerEmbedder(EmbeddingInterface):
    """Uses a local HuggingFace Sentence Transformer model."""
    def __init__(self, model_name: str = config.EMBEDDER_MODELS["default"]):
        try:
            self.model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'} # Use 'cuda' if available
            )
            print(f"Loaded SentenceTransformer: {model_name}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)

class GoogleEmbedder(EmbeddingInterface):
    """Uses the Google Generative AI embedding API."""
    def __init__(self, model_name: str = config.EMBEDDER_MODELS["google"]):
        try:
            self.model = GoogleGenerativeAiEmbeddings(
                model=model_name, 
                google_api_key=config.GOOGLE_API_KEY
            )
            print(f"Loaded GoogleEmbedder: {model_name}")
        except Exception as e:
            print(f"Error loading GoogleGenerativeAiEmbeddings: {e}")
            print("Please ensure GOOGLE_API_KEY is set correctly in your .env file.")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)

# --- Factory Function ---

def get_embedder() -> EmbeddingInterface:
    """Factory function to get the configured embedder."""
    if config.EMBEDDER_PROVIDER == "google":
        return GoogleEmbedder()
    elif config.EMBEDDER_PROVIDER == "default":
        return SentenceTransformerEmbedder()
    else:
        raise ValueError(f"Unknown embedder provider: {config.EMBEDDER_PROVIDER}")
