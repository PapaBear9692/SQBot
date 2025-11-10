from flask import request, jsonify
from model.ragModel import RAGModel

# --- Initialization ---
try:
    rag_model = RAGModel()
    print("RAGModel initialized successfully.")
except Exception as e:
    rag_model = None
    print(f" Failed to initialize RAGModel.\nError: {e}")

# --- Route Handler ---
def handle_chat_request():
    """
    Handles /ask POST requests.
    Retrieves answer using RAGModel (with hybrid search) and returns
    the final answer, sources, and reranked candidates.
    """
    if not rag_model:
        return jsonify({
            "answer": "RAG system failed to initialize. Check server logs.",
            "status": "error"
        }), 500

    data = request.get_json(silent=True)
    user_question = data.get("prompt") if data else None

    if not user_question or not user_question.strip():
        return jsonify({
            "answer": "No prompt provided.",
            "status": "error"
        }), 400

    print(f"Received question: {user_question}")

    try:
        # Run retrieval + reasoning
        answer, final_ranked = rag_model.generate_answer(user_question)
        print(answer)

        # Extract a simplified sources list for frontend
        sources = [
            {
                "text": doc.get("text", ""),
                "summary": doc.get("summary", ""),
                "topic": doc.get("topic", ""),
                "source": doc.get("source", ""),
                "cosine_score": doc.get("cosine_score", None),
                "rerank_score": doc.get("rerank_score", None)
            }
            for doc in final_ranked
        ]

        return jsonify({
            "status": "success",
            "answer": answer,
            "sources": sources,
            "reranked": final_ranked  # full details if needed
        })

    except Exception as e:
        print(f"Error in handle_chat_request: {e}")
        return jsonify({
            "answer": "An internal error occurred during RAG processing.",
            "status": "error"
        }), 500
