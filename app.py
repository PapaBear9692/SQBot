import os
import secrets

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from app_config import init_settings_and_storage
from prompt import PROMPT_TEMPLATE, CONDENSE_PROMPT


load_dotenv()

text_qa_template = PromptTemplate(PROMPT_TEMPLATE)
condense_prompt = PromptTemplate(CONDENSE_PROMPT)

app = Flask(__name__, static_url_path="/chat/static")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)


storage_context = init_settings_and_storage()
vector_store = storage_context.vector_store
index = VectorStoreIndex.from_vector_store(vector_store)
chat_memories = {}
chat_engines = {}

def get_chat_engine(conv_id: str):
    if not conv_id:
        conv_id = "default"

    if conv_id not in chat_memories:
        chat_memories[conv_id] = ChatMemoryBuffer.from_defaults(token_limit=1024)
        chat_engines[conv_id] = index.as_chat_engine(
            chat_mode="condense_question", # type: ignore
            memory=chat_memories[conv_id],
            text_qa_template=text_qa_template,
            similarity_top_k=7,
            response_mode="compact",
            condense_question_promp=condense_prompt,
        )
    return chat_engines[conv_id]



@app.route("/")
def index_route():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat_route():
    user_msg = (request.form.get("msg") or "").strip()
    conv_id = (request.form.get("conversation_id") or "").strip()
    if not user_msg:
        return jsonify({"response": "Please enter a question."})

    try:
        chat_engine = get_chat_engine(conv_id)
        response_obj = chat_engine.chat(user_msg)
        answer = (str(response_obj) if response_obj is not None else "").strip()
        if not answer:
            answer = "I could not generate a response."

        #sources = []
        # for sn in getattr(response_obj, "source_nodes", []) or []:
        #     node = sn.node
        #     meta = getattr(node, "metadata", {}) or {}
        #     sources.append(
        #         {
        #             "score": float(getattr(sn, "score", 0.0) or 0.0),
        #             "product_name": meta.get("product_name"),
        #             "usage": meta.get("usage"),
        #             "file_name": meta.get("file_name"),
        #             "normalized_name": meta.get("normalized_name"),
        #         }
        #     )

    except Exception as e:
        print(f"Error in /get handler: {e!r}")
        answer = "An error occurred while generating a response."
        #sources = []

    return jsonify({"response": answer, "conversation_id": conv_id})


@app.route("/reset", methods=["POST"])
def reset_conversation():
    conv_id = (request.form.get("conversation_id") or "").strip()
    if conv_id in chat_memories:
        chat_memories.pop(conv_id, None)
        chat_engines.pop(conv_id, None)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=False)

