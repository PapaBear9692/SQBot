import config
from core.loaders_and_splitters import get_loader, get_splitter
from core.embedders import get_embedder
from core.vector_stores import get_vector_store

def main():
    """
    Main function to run the data pipeline:
    1. Load data
    2. Split data
    3. Embed data
    4. Upsert data to vector store
    """
    print("--- Starting Data Pipeline ---")
    
    # 1. Get configured components from factories
    print(f"Using Embedder: {config.EMBEDDER_PROVIDER}")
    print(f"Using LLM: {config.LLM_PROVIDER}")
    print(f"Using Vector Store: Pinecone (Index: {config.PINECONE_INDEX_NAME})")
    
    loader = get_loader()
    splitter = get_splitter()
    embedder = get_embedder()
    vector_store = get_vector_store()
    
    # 2. Load Documents
    documents = loader.load(config.DATA_DIRECTORY)
    if not documents:
        print(f"No documents found in {config.DATA_DIRECTORY}. Exiting pipeline.")
        return
    print(f"Loaded {len(documents)} documents.")
    
    # 3. Split Documents
    chunks = splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    
    # 4. Embed Documents
    # Get text content from Document objects
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    print("Embedding documents... (This may take a while)")
    embeddings = embedder.embed_documents(chunk_texts)
    print(f"Created {len(embeddings)} embeddings.")
    
    # 5. Upsert to Vector Store
    # Pass the original Document objects (chunks) to retain metadata
    print("Upserting documents to Pinecone...")
    vector_store.upsert_documents(chunks, embeddings)
    
    print("--- Data Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
