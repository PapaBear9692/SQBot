import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Model Selection (THE "SWITCH") ---
# Change these values to swap models easily

# Supported: "gemini", "openai"
LLM_PROVIDER = "gemini"

# Supported: "default" (all-miniLm-l6), "google" (text-embedder-small)
EMBEDDER_PROVIDER = "default"

# --- Model Names ---
# You can also change these if new models are released
LLM_MODELS = {
    "gemini": "gemini-1.5-flash", # Updated from 2.5
    "openai": "gpt-3.5-turbo"
}

EMBEDDER_MODELS = {
    "default": "all-MiniLM-L6-v2",
    "google": "models/text-embedding-004" # Updated from small
}

# --- Vector Store Config ---
PINECONE_INDEX_NAME = "medical-rag-index" # <<< IMPORTANT: Change this to your index name
# The dimension must match your chosen embedder
# all-MiniLM-L6-v2 -> 384
# text-embedding-004 -> 768
EMBEDDING_DIMENSIONS = {
    "default": 384,
    "google": 768
}

def get_current_embedding_dimension():
    """Helper function to get the dimension for the currently selected embedder."""
    return EMBEDDING_DIMENSIONS.get(EMBEDDER_PROVIDER, 384)

# --- Data Processing Config ---
DATA_DIRECTORY = "./data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- RAG Config ---
TOP_K_RESULTS = 3
