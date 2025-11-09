import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from model.embedderModel import get_embedder
from model.storageModel import PineconeStorage
from utils.app_config import EMBEDDER_PROVIDER


def hybrid_search(query: str, top_k: int = 10, rerank_top: int = 5):
    """
    Hybrid search combining:
    1️⃣ Pinecone vector retrieval (fast, approximate)
    2️⃣ Local cosine similarity re-ranking (precise)
    3️⃣ Cross-encoder LLM reranking (semantic fine-tuning)
    """
    print(f"\n--- Running Hybrid Search for Query: '{query}' ---")

    # 1. Initialize components
    embedder = get_embedder(EMBEDDER_PROVIDER)
    storage = PineconeStorage()

    # 2. Embed query
    print("Embedding query...")
    query_vector = embedder.embed_query(query)

    # 3. Retrieve top-k from Pinecone
    print(f"Fetching top {top_k} candidates from Pinecone...")
    results = storage.query_vectors(query_vector, top_k=top_k)

    if not results.get("matches"):
        print("No matches found.")
        return []

    # 4. Local cosine similarity refinement
    print("Refining candidates locally using cosine similarity...")
    local_ranked = []
    for match in results["matches"]:
        candidate_vector = np.array(match["values"])
        local_score = cosine_similarity([query_vector], [candidate_vector])[0][0]
        local_ranked.append({
            "id": match["id"],
            "text": match["metadata"].get("text", ""),
            "summary": match["metadata"].get("summary", ""),
            "topic": match["metadata"].get("topic", ""),
            "source": match["metadata"].get("source", ""),
            "cosine_score": float(local_score),
        })

    # Sort by local cosine similarity
    local_ranked = sorted(local_ranked, key=lambda x: x["cosine_score"], reverse=True)
    top_local = local_ranked[:rerank_top]

    # 5. Cross-encoder reranking for final precision
    print("Running cross-encoder reranking (semantic scoring)...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(query, doc["text"]) for doc in top_local]
    scores = reranker.predict(pairs)

    for doc, score in zip(top_local, scores):
        doc["rerank_score"] = float(score)

    # Sort final list by LLM rerank score
    final_ranked = sorted(top_local, key=lambda x: x["rerank_score"], reverse=True)

    # 6. Display top results
    print("\nTop Results after Hybrid Reranking:\n")
    for i, doc in enumerate(final_ranked, start=1):
        print(f"🔹 Rank {i} | Score: {doc['rerank_score']:.4f} | Cosine: {doc['cosine_score']:.4f}")
        print(f"Topic: {doc['topic']}")
        print(f"Summary: {doc['summary']}")
        print(f"Source: {doc['source']}")
        print(f"Text Preview: {doc['text'][:200]}...\n")

    return final_ranked
