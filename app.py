import os
import secrets

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response

from llama_index.core import VectorStoreIndex
from app_config import init_settings_and_storage
from router import handle_chat_message, handle_chat_message_sse, reset_conversation

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

@app.route("/stream", methods=["POST"])
def stream_route():
    """
    SSE streaming endpoint.
    Frontend should read an EventSource-like stream and append tokens.
    """
    user_msg = (request.form.get("msg") or "").strip()
    conv_id = (request.form.get("conversation_id") or "").strip() or "default"

    if not user_msg:
        def _one():
            yield "event: message\n"
            yield "data: Please enter a question.\n\n"
            yield "event: done\n"
            yield "data: {}\n\n"
        return Response(_one(), mimetype="text/event-stream")

    def event_stream():
        try:
            yield from handle_chat_message_sse(index, user_msg, conv_id)
        except Exception as e:
            print(f"Error in /stream handler: {e!r}")
            yield "event: error\n"
            yield "data: An error occurred while generating a response.\n\n"
            yield "event: done\n"
            yield "data: {}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(event_stream(), mimetype="text/event-stream", headers=headers)


@app.route("/reset", methods=["POST"])
def reset_route():
    conv_id = (request.form.get("conversation_id") or "").strip() or "default"
    reset_conversation(conv_id)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
