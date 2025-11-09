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
        # Run full retrieval + reasoning
        result = rag_model.generate_answer(user_question)

        # Unpack
        answer = result.get("answer")
        sources = result.get("sources", [])
        reranked = result.get("reranked", [])

        # Return both the final answer and sources (optional)
        return jsonify({
            "status": "success",
            "answer": answer,
            "sources": sources,
            "reranked": reranked
        })

    except Exception as e:
        print(f"❌ Error in handle_chat_request: {e}")
        return jsonify({
            "answer": "An internal error occurred during RAG processing.",
            "status": "error"
        }), 500
