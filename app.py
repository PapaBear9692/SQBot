import os
from flask import Flask, render_template, request, jsonify
from controller.appController import handle_chat_request
from utils.app_config import LLM_PROVIDER, EMBEDDER_PROVIDER

# --- Flask App Setup ---

# Determine the template directory path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- Routes ---

@app.route('/')
def index():
    # Pass model info to the frontend
    return render_template(
        'chat.html',
        llm_provider=LLM_PROVIDER,
        embedder_provider=EMBEDDER_PROVIDER
    )

@app.route('/ask', methods=['POST'])
def ask():
    return handle_chat_request()


# --- Main Execution ---
if __name__ == '__main__':
    print("--- SQBot RAG Application Starting ---")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Embedder Provider: {EMBEDDER_PROVIDER}")
    app.run(debug=True, port=5000)