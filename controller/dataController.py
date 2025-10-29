import uuid
from pathlib import Path
from model.dataLoaderModel import DataLoader
from model.embedderModel import get_embedder
from model.storageModel import PineconeStorage
from utils.app_config import EMBEDDER_PROVIDER
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def run_indexing_pipeline():
    """
    Orchestrates the data loading, splitting, embedding, and uploading process.
    This function is intended to be run by a separate script (e.g., run_indexing.py).
    """
    print("--- Starting Data Indexing Pipeline ---")
    
    try:
        # 1. Initialize Components
        print("Initializing components...")
        data_loader = DataLoader()
        embedder = get_embedder(EMBEDDER_PROVIDER)
        storage = PineconeStorage()

        # 2. Load and Split Documents
        print(f"Loading documents from '{data_loader.data_dir}'...")
        documents = []
        for file in Path(data_loader.data_dir).rglob("*"):
            if file.suffix.lower() == ".pdf":
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

        # 4. Prepare Vectors for Upsert
        print("Preparing vectors for upsert...")
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        print(f"Embedding {len(chunk_texts)} chunks... (This may take a while)")
        embeddings = embedder.embed_documents(chunk_texts)
        print("Embedding complete.")
        
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = str(uuid.uuid4())  # Generate a unique ID
            metadata = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": int(chunk.metadata.get("page", 0))
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
