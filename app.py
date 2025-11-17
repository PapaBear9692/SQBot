import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from controller.appController import rag_pipeline

# --- Flask App Setup ---

# Determine the template directory path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Load .env file
load_dotenv()


app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- Routes ---

@app.route('/')
def index():
    # Pass model info to the frontend
    return render_template(
        'chat.html',
        llm_provider="Google",
        embedder_provider="PubmedBERT"
    )

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_q = data.get("prompt", "").strip()

    if not user_q:
        return jsonify({"error": "Query missing"}), 400

    answer = rag_pipeline(user_q)
    return jsonify({"answer": answer})


# --- Main Execution ---
if __name__ == '__main__':
    print("--- SQBot RAG Application Starting ---")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Embedder Provider: {EMBEDDER_PROVIDER}")
    app.run(debug=True, port=5000)