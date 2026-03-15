import os
import secrets

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response

from llama_index.core import VectorStoreIndex
from app_config import init_settings_and_storage
from router import handle_chat_message, handle_chat_message_stream, reset_conversation

load_dotenv()

# ----------------------------
# App + Index setup
# ----------------------------
app = Flask(__name__, static_url_path="/ai/static")
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
        answer, cid, intent = handle_chat_message(index, user_msg, conv_id)
    except Exception as e:
        print(f"Error in /get handler: {e!r}")
        answer, cid, intent = "An error occurred while generating a response.", conv_id, None

    return jsonify({"response": answer, "conversation_id": cid, "intent": intent})



@app.route("/reset", methods=["POST"])
def reset_route():
    conv_id = (request.form.get("conversation_id") or "").strip() or "default"
    reset_conversation(conv_id)
    return jsonify({"status": "ok"})


@app.route("/stream", methods=["POST"])
def stream_route():
    """Streaming endpoint using Server-Sent Events (SSE)."""
    import json
    import traceback
    
    user_msg = (request.form.get("msg") or "").strip()
    conv_id = (request.form.get("conversation_id") or "").strip() or "default"

    if not user_msg:
        return jsonify({"error": "Please enter a question.", "conversation_id": conv_id}), 400

    def generate():
        try:
            #print(f"Stream handler: Processing msg='{user_msg[:50]}' for conv_id='{conv_id}'")
            token_gen, cid, intent = handle_chat_message_stream(index, user_msg, conv_id)
            
            # Send initial metadata as first event
            start_event = json.dumps({"type": "start", "conversation_id": cid, "intent": intent})
            yield f"data: {start_event}\n\n"
            
            token_count = 0
            # Stream tokens
            for token in token_gen:
                token_count += 1
                token_event = json.dumps({"type": "token", "content": token})
                yield f"data: {token_event}\n\n"
            
            print(f"Stream handler: Sent {token_count} tokens")
            
            # Send end event
            end_event = json.dumps({"type": "end", "conversation_id": cid})
            yield f"data: {end_event}\n\n"
            
        except Exception as e:
            print(f"Error in /stream handler: {e!r}")
            print(traceback.format_exc())
            error_msg = f"An error occurred: {str(e)[:100]}"
            error_event = json.dumps({"type": "error", "content": error_msg})
            yield f"data: {error_event}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5005)
