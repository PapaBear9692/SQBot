from typing import Tuple, List, Dict, Any
import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from model.embedderModel import get_embedder
from model.llmModel import get_llm
from model.storageModel import PineconeStorage
from utils.app_config import EMBEDDER_PROVIDER, TOP_K_RESULTS

class RAGModel:
    """
    RAGModel with hybrid retrieval:
    1️⃣ Pinecone top-k
    2️⃣ Local cosine similarity re-ranking
    3️⃣ Cross-encoder reranking
    """
    def __init__(self):
        print("Initializing RAGModel (hybrid search)...")
        self.embedder = get_embedder(EMBEDDER_PROVIDER)
        self.llm = get_llm()
        self.vector_store = PineconeStorage()
        self.vector_store.connect_to_index()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("RAGModel initialized successfully.")

    def generate_answer(self, prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        try:
            # 1️⃣ Embed query
            print(f"Embedding query: '{prompt}'")
            query_vector = self.embedder.embed_query(prompt)

            # 2️⃣ Retrieve top-k from Pinecone
            print(f"Fetching top {TOP_K_RESULTS} candidates from Pinecone...")
            context_docs = self.vector_store.query(query_vector, top_k=TOP_K_RESULTS)

            if not context_docs:
                print("No matches found in Pinecone.")
                return "I could not find relevant information.", []

            # 3️⃣ Local cosine similarity reranking
            print("Refining candidates with local cosine similarity...")
            local_ranked = []
            for doc in context_docs:
                candidate_vector = np.array(doc.get("embedding", []))
                if candidate_vector.size == 0:
                    continue
                cos_score = cosine_similarity([query_vector], [candidate_vector])[0][0]
                local_ranked.append({
                    **doc,
                    "cosine_score": float(cos_score)
                })
            local_ranked.sort(key=lambda x: x["cosine_score"], reverse=True)
            top_local = local_ranked[:TOP_K_RESULTS]

            # 4️⃣ Cross-encoder reranking
            print("Applying cross-encoder reranking...")
            pairs = [(prompt, doc.get("text", "")) for doc in top_local]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(top_local, scores):
                doc["rerank_score"] = float(score)
            final_ranked = sorted(top_local, key=lambda x: x["rerank_score"], reverse=True)

            # 5️⃣ Call LLM with top context
            print("Generating final answer from LLM...")
            llm_context = [{"text": doc.get("text", ""), "summary": doc.get("summary", ""), "topic": doc.get("topic", "")} for doc in final_ranked]
            answer = self.llm.generate_response(prompt=prompt, context=llm_context)

            return answer, final_ranked

        except Exception as e:
            print(f"Error in RAGModel.generate_answer: {e}")
            return "An error occurred while processing your request.", []
