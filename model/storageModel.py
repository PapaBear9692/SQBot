from typing import List
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from core.interfaces import VectorStoreInterface
import config

class PineconeVectorStore(VectorStoreInterface):
    """Implementation for Pinecone vector store."""
    def __init__(self, index_name: str = config.PINECONE_INDEX_NAME):
        try:
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index_name = index_name
            self._create_index_if_not_exists()
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            print("Please ensure PINECONE_API_KEY and PINECONE_INDEX_NAME are set correctly.")
            raise
    
    def _create_index_if_not_exists(self):
        """Creates the Pinecone index if it doesn't already exist."""
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}' in Pinecone...")
            try:
                dimension = config.get_current_embedding_dimension()
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine", # Cosine similarity is common for text
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Index '{self.index_name}' created successfully with dimension {dimension}.")
            except Exception as e:
                print(f"Error creating Pinecone index: {e}")
                raise
        else:
            print(f"Index '{self.index_name}' already exists.")

    def upsert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Upserts documents and their embeddings."""
        vectors_to_upsert = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            vector_id = f"doc_{i}" # Simple ID, consider a more robust strategy
            vectors_to_upsert.append({
                "id": vector_id,
                "values": emb,
                "metadata": {
                    "text": doc.page_content,
                    **doc.metadata # Add metadata like source file
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            try:
                print(f"Upserting batch {i//batch_size + 1}...")
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error upserting batch: {e}")
        print("Upsert complete.")

    def query(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """Queries the vector store."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Convert Pinecone results back to LangChain Document objects
            documents = []
            for match in results.get('matches', []):
                doc = Document(
                    page_content=match.get('metadata', {}).get('text', ''),
                    metadata={k: v for k, v in match.get('metadata', {}).items() if k != 'text'}
                )
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

# --- Factory Function ---

def get_vector_store() -> VectorStoreInterface:
    """Factory function to get the configured vector store."""
    # Currently only supports Pinecone, but you could add more
    return PineconeVectorStore()
