import os
import secrets

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from llama_index.core import VectorStoreIndex
from app_config import init_settings_and_storage
from router import handle_chat_message, reset_conversation

load_dotenv()

# ----------------------------
# App + Index setup
# ----------------------------
app = Flask(__name__, static_url_path="/chat/static")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)

storage_context = init_settings_and_storage()
vector_store = storage_context.vector_store
index = VectorStoreIndex.from_vector_store(vector_store)

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index_route():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat_route():
    """Non-streaming (legacy) endpoint."""
    user_msg = (request.form.get("msg") or "").strip()
    conv_id = (request.form.get("conversation_id") or "").strip() or "default"

    if not user_msg:
        return jsonify({"response": "Please enter a question.", "conversation_id": conv_id})

    try:
        answer, cid = handle_chat_message(index, user_msg, conv_id)
    except Exception as e:
        print(f"Error in /get handler: {e!r}")
        answer, cid = "An error occurred while generating a response.", conv_id

    return jsonify({"response": answer, "conversation_id": cid})

@app.route("/reset", methods=["POST"])
def reset_route():
    conv_id = (request.form.get("conversation_id") or "").strip() or "default"
    reset_conversation(conv_id)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
