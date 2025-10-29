from typing import List
from model.core_interface import EmbeddingInterface
from langchain_huggingface import HuggingFaceEmbeddings


from utils.app_config import (
    EMBEDDER_PROVIDER,
    EMBEDDER_MODELS,
    EMBEDDING_DIMENSIONS,
    GOOGLE_API_KEY
)


class SentenceTransformerEmbedder(EmbeddingInterface):
    """Uses a local HuggingFace Sentence Transformer model."""
    def __init__(self):
        self.model_name = EMBEDDER_MODELS["default"]
        self.dimension = EMBEDDING_DIMENSIONS["default"]
        try:
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'}  # Use 'cuda' if available
            )
            print(f"Loaded SentenceTransformer: {self.model_name}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            raise

    def get_dimension(self) -> int:
        return self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)


class GoogleEmbedder(EmbeddingInterface):
    """Uses the Google Generative AI embedding API."""
    def __init__(self):
        self.model_name = EMBEDDER_MODELS["google"]
        self.dimension = EMBEDDING_DIMENSIONS["google"]
        try:
            self.model = GoogleGenerativeAiEmbeddings(
                model=self.model_name,
                google_api_key=GOOGLE_API_KEY
            )
            print(f"Loaded GoogleEmbedder: {self.model_name}")
        except Exception as e:
            print(f"Error loading GoogleGenerativeAiEmbeddings: {e}")
            print("Please ensure GOOGLE_API_KEY is set correctly.")
            raise

    def get_dimension(self) -> int:
        return self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)


# --- Factory Function ---

def get_embedder(provider: str = EMBEDDER_PROVIDER) -> EmbeddingInterface:
    """
    Factory function to get the configured embedder
    based on the 'EMBEDDER_PROVIDER' in app_config.py.
    """
    if provider == "google":
        return GoogleEmbedder()
    elif provider == "default":
        return SentenceTransformerEmbedder()
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")
