from flask import request, jsonify
from model.ragModel import RAGModel

# --- Initialization ---
try:
    rag_model = RAGModel()
    print("appController: RAGModel initialized successfully.")
except Exception as e:
    rag_model = None
    print(f"FATAL: appController: RAGModel failed to initialize. Error: {e}")

# --- Route Handler ---

def handle_chat_request():
    # Check if the RAG model loaded correctly
    if not rag_model:
        return jsonify({
            'answer': "Error: The RAG system is not initialized. Please check the server logs."
        }), 500

    # Get JSON data from the request
    data = request.get_json()
    user_question = data.get('prompt')
    print(f"appController: Received prompt: {user_question}")

    if not user_question:
        return jsonify({'answer': "Error: No prompt provided."}), 400

    try:
        # 1. Get the answer from the RAG model
        answer, context_docs = rag_model.generate_answer(user_question)
        print(f"Generated answer: {answer}")
        for i, doc in enumerate(context_docs):
            print(f"Context Doc {i+1}: Score={doc['score']}, Text={doc['text'][:100]}...")
            print(f"----------")
        # 2. Return the answer
        return jsonify({'answer': answer})
        
    except Exception as e:
        print(f"Error in appController.handle_chat_request: {e}")
        return jsonify({'answer': "An error occurred while processing your request."}), 500