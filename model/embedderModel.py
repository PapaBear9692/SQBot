from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
from utils.app_config import EMBEDDER_MODELS, EMBEDDING_DIMENSIONS, OPENAI_API_KEY


class OpenSourceEmbedder():
    def __init__(self, model_name, dimension):
        # LangChain's SentenceTransformerEmbeddings handles the model loading
        self.embedder = SentenceTransformerEmbeddings(
            model_name=model_name
        )
        self._dimension = dimension

    def embed_documents(self, texts):
        return self.embedder.embed_documents(texts)

    def embed_query(self, text):
        return self.embedder.embed_query(text)
        
    def get_dimension(self):
        return self._dimension


class OpenAIEmbedder():
    def __init__(self, model_name, dimension, api_key):
        if not api_key:
            raise ValueError("OpenAI API Key not found.")
            
        self.embedder = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
        self._dimension = dimension

    def embed_documents(self, texts):
        # OpenAI returns embeddings directly as a list of lists (embeddings for documents)
        return self.embedder.embed_documents(texts)

    def embed_query(self, text):
        # OpenAI returns a single embedding for the query
        return self.embedder.embed_query(text)
        
    def get_dimension(self):
        return self._dimension


def get_embedder(provider: str):
    if provider == "default":
        model_name = EMBEDDER_MODELS["default"]
        dimension = EMBEDDING_DIMENSIONS["default"]
        print(f"Using default HuggingFace Embedder: {model_name}")
        return OpenSourceEmbedder(model_name, dimension)
    
    # Logic to support the PubMedBert model
    elif provider == "PubMedBert":
        model_name = EMBEDDER_MODELS["PubMedBert"]
        dimension = EMBEDDING_DIMENSIONS["PubMedBert"]
        print(f"Using PubMedBert Embedder: {model_name}")
        return OpenSourceEmbedder(model_name, dimension)

    elif provider == "openai":
        model_name = EMBEDDER_MODELS["openai"]
        dimension = EMBEDDING_DIMENSIONS["openai"]
        print(f"Using OpenAI Embedder: {model_name}")
        return OpenAIEmbedder(model_name, dimension, OPENAI_API_KEY)
    
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")
