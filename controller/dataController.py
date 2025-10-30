import uuid
import json
from pathlib import Path
from model.dataLoaderModel import DataLoader
from model.embedderModel import get_embedder
from model.storageModel import PineconeStorage
from utils.app_config import EMBEDDER_PROVIDER
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Define the path to the product map file
PRODUCT_MAP_PATH = "utils/product_page_map.json"


def run_indexing_pipeline():
    """
    Orchestrates the data loading, splitting, embedding, and uploading process.
    This function is intended to be run by a separate script (e.g., run_indexing.py).
    """
    print("--- Starting Data Indexing Pipeline ---")
    
    try:
        # 1. Initialize Components and Load Map
        print("Initializing components...")
        data_loader = DataLoader()
        embedder = get_embedder(EMBEDDER_PROVIDER)
        storage = PineconeStorage()

        # Load the product map JSON file
        try:
            with open(PRODUCT_MAP_PATH, 'r') as f:
                product_map = json.load(f)
            print(f"Successfully loaded product map from {PRODUCT_MAP_PATH}.")
        except FileNotFoundError:
            print(f"ERROR: Product map file not found at {PRODUCT_MAP_PATH}. Using empty map.")
            product_map = {}
        except json.JSONDecodeError:
            print(f"ERROR: Failed to parse JSON from {PRODUCT_MAP_PATH}. Using empty map.")
            product_map = {}

        # 2. Load and Split Documents
        print(f"Loading documents from '{data_loader.data_dir}'...")
        documents = []
        for file in Path(data_loader.data_dir).rglob("*"):
            if file.suffix.lower() == ".pdf":
                # Ensure LangChain's page metadata starts at 0 for PDF
                documents.extend(PyPDFLoader(str(file)).load())
            elif file.suffix.lower() == ".txt":
                documents.extend(TextLoader(str(file)).load())

        if not documents:
            print(f"No documents found in '{data_loader.data_dir}'. Aborting.")
            return

        print(f"Loaded {len(documents)} document pages/sources.")
        chunks = data_loader.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")

        if not chunks:
            print("No chunks were created. Aborting.")
            return

        # 3. Create Pinecone Index (if it doesn't exist)
        dimension = embedder.get_dimension()
        print(f"Ensuring Pinecone index exists (Dimension: {dimension})...")
        storage.create_index_if_not_exists(dimension)

        # 4. Prepare Vectors for Upsert (with ENRICHED metadata)
        print("Preparing and embedding vectors for upsert...")
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        # Embed all chunk texts
        print(f"Embedding {len(chunk_texts)} chunks... (This may take a while)")
        embeddings = embedder.embed_documents(chunk_texts) 
        print("Embedding complete.")
        
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = str(uuid.uuid4())
            
            # 1. Get the page number and adjust for 1-based indexing
            # LangChain's PDF Loader uses 0-based indexing for "page" metadata
            page_num_int = chunk.metadata.get("page", -1)
            if isinstance(page_num_int, int) and page_num_int >= 0:
                page_num_int += 1
            else:
                # If page metadata is missing or invalid, treat it as unknown
                page_num_int = 0

            page_num_str = str(page_num_int)

            # 2. METADATA ENRICHMENT: Look up the product name
            # Uses the 1-based page number string as the key
            product_name = product_map.get(page_num_str, "Unknown")
            
            # 3. Create the final metadata dictionary
            metadata = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": page_num_int,
                "product_name": product_name 
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

        # 5. Upsert to Pinecone
        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        storage.upsert_vectors(vectors_to_upsert)
        
        print("--- Data Indexing Pipeline Finished Successfully ---")

    except Exception as e:
        print(f"FATAL ERROR during indexing: {e}")
        import traceback
        traceback.print_exc()
