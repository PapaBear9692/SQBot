# app_config.py
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

# cache lines
CACHE_DIR = ROOT_DIR / "model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "abhinand/MedEmbed-base-v0.1"
EMBEDDING_DIM = 768

#"llama-3.1-8b-instant" #"qwen/qwen3-32b" #"llama-3.3-70b-versatile" #"models/gemini-2.5-flash" #"models/gemini-2.5-flash-lite"
LLM_MODEL_NAME =  "models/gemini-2.5-flash-lite"

PINECONE_INDEX_NAME = "medicine-chatbot-llamaindex-medembed"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_NAMESPACE = None


def init_settings_and_storage():
    """Initialize embeddings, LLM, automatic chunking, and Pinecone vector store."""
    # Load environment variables
    load_dotenv(ENV_PATH)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not pinecone_api_key or not google_api_key or not groq_api_key:
        raise ValueError("Missing PINECONE_API_KEY or GOOGLE_API_KEY in .env")

    # Embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device="cpu",  # change to "cuda" if you have GPU
        cache_folder=str(CACHE_DIR),
    )

    # LLM
    if(LLM_MODEL_NAME == "models/gemini-2.5-flash" or LLM_MODEL_NAME == "models/gemini-2.5-flash-lite"):
        Settings.llm = GoogleGenAI(
            model=LLM_MODEL_NAME,
            api_key=google_api_key,
            temperature=0.5,
        )
    else:
        Settings.llm = Groq(
            model=LLM_MODEL_NAME,
            api_key=groq_api_key,
            temperature=0.5,
        )


    # Pinecone setup
    pc = Pinecone(api_key=pinecone_api_key)

    # Depending on your pinecone client version, list_indexes() typically
    # returns an object with a .names() method that lists index names.
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
    else:
        print(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'")

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=PINECONE_NAMESPACE,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

