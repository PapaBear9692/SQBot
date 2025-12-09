import os
import secrets

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from app_config import init_settings_and_storage
from prompt import PROMPT_TEMPLATE

load_dotenv()

text_qa_template = PromptTemplate(PROMPT_TEMPLATE)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)


storage_context = init_settings_and_storage()
vector_store = storage_context.vector_store

index = VectorStoreIndex.from_vector_store(vector_store)
memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",  # rewrites follow-ups like "dosage?" into full questions
    memory=memory,
    text_qa_template=text_qa_template,
    similarity_top_k=7,
    response_mode="compact",
)

@app.route("/")
def index_route():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat_route():
    user_msg = (request.form.get("msg") or "").strip()
    if not user_msg:
        return jsonify({"response": "Please enter a question."})

    try:

        response_obj = chat_engine.chat(user_msg)
        answer = (str(response_obj) if response_obj is not None else "").strip()
        if not answer:
            answer = "I could not generate a response."

        sources = []
        for sn in getattr(response_obj, "source_nodes", []) or []:
            node = sn.node
            meta = getattr(node, "metadata", {}) or {}
            sources.append(
                {
                    "score": float(getattr(sn, "score", 0.0) or 0.0),
                    "product_name": meta.get("product_name"),
                    "usage": meta.get("usage"),
                    "file_name": meta.get("file_name"),
                    "normalized_name": meta.get("normalized_name"),
                }
            )

    except Exception as e:
        print(f"Error in /get handler: {e!r}")
        answer = "An error occurred while generating a response."
        sources = []

    return jsonify({"response": answer, "sources": sources})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
