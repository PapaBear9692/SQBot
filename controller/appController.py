from flask import Flask, render_template, request, jsonify
from rag_service import rag_service_instance # Import the singleton
import config

app = Flask(__name__)

# --- Controller (Routes) ---

@app.route("/")
def index():
    """
    Serves the main chat page (the "View").
    """
    return render_template("index.html",
                           llm_provider=config.LLM_PROVIDER,
                           embedder_provider=config.EMBEDDER_PROVIDER
                           )

@app.route("/ask", methods=["POST"])
def ask():
    """
    API endpoint to handle user prompts (the "Prompt Collector").
    It calls the "Model" (RAGService) to get an answer.
    """
    if not rag_service_instance:
        return jsonify({"error": "RAGService failed to initialize. Check server logs."}), 500
        
    data = request.get_json()
    prompt = data.get("prompt")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
        
    print(f"Received prompt: {prompt}")
    
    # 1. Call the RAG service to get the answer
    answer = rag_service_instance.get_answer(prompt)
    
    # 2. Return the answer as JSON
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
