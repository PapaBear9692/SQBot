from typing import List
from pinecone.grpc import PineconeGRPC as Pinecone
from utils.app_config import PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K_RESULTS

class PineconeStorage:
    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment.")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.index = self.pc.Index(self.index_name)
        print(f"PineconeStorage initialized for index: '{self.index_name}")

    def query(self, query_embedding: List[float]):
        try:
            response = self.index.query(
            top_k= TOP_K_RESULTS,                     # <-- required
            vector=query_embedding,
            include_metadata=True,
            include_values=False
        )


            results = []
            for match in response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {},
                    "text": match.metadata.get("text", "") if match.metadata else ""
                })

            return results

        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
