from flask import request, jsonify
from model.ragModel import RAGModel

# --- Initialization ---

# Initialize the RAG Model. This runs once when the app starts.
# It pre-loads the LLM, Embedder, and connects to Pinecone.
try:
    rag_model = RAGModel()
    print("appController: RAGModel initialized successfully.")
except Exception as e:
    rag_model = None
    print(f"FATAL: appController: RAGModel failed to initialize. Error: {e}")

# --- Route Handler ---

def handle_chat_request():
    """
    Handles the POST request from the /ask API endpoint.
    It gets the user's question, passes it to the RAG model,
    and returns the generated answer.
    """
    # Check if the RAG model loaded correctly
    if not rag_model:
        return jsonify({
            'answer': "Error: The RAG system is not initialized. Please check the server logs."
        }), 500

    # Get JSON data from the request
    data = request.get_json()
    user_question = data.get('prompt')

    if not user_question:
        return jsonify({'answer': "Error: No prompt provided."}), 400

    print(f"appController: Received prompt: {user_question}")

    try:
        # 1. Get the answer from the RAG model
        answer, sources = rag_model.generate_answer(user_question)
        
        # 2. Return the answer
        # You can also return 'sources' if you want to display them on the frontend
        return jsonify({'answer': answer})
        
    except Exception as e:
        print(f"Error in appController.handle_chat_request: {e}")
        return jsonify({'answer': "An error occurred while processing your request."}), 500