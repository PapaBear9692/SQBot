import os
import secrets
import re

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

# Track last user message per conversation (for context switching)
last_user_msgs = {}  # conv_id -> last user message

# Track last product mentioned by the bot (for pronoun/generic follow-ups like "its dosage")
last_product_mentioned = {}  # conv_id -> product name string


# ----------------------------
# Context switch helpers
# ----------------------------

FOLLOWUP_GENERIC = {
    "dosage", "dose", "dose?", "dosage?", "what is the dosage", "what is its dosage",
    "side effects", "side effect", "side effects?", "warning", "warnings", "how to use",
    "usage", "contraindications", "contraindication", "precautions", "interaction", "interactions",
    "price", "adult dose", "child dose", "children dose", "kids dose"
}

LIST_TRIGGERS = {
    "product list", "all product", "all products", "available products",
    "show all", "list all", "full list", "all medicine", "all medicines",
    "catalog", "catalogue"
}

GREETINGS = {
    "hi", "hello", "hey", "good morning", "good evening", "good night",
    "thanks", "thank you", "ok", "okay"
}


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str):
    return set(re.findall(r"[a-zA-Z0-9]+", (s or "").lower()))


def extract_product_name_from_answer(text: str):
    """
    Lightweight heuristic: grabs the first TitleCase token >= 4 chars.
    Works well for typical medicine/product names like 'Tryptin', 'Revatol', etc.
    """
    if not text:
        return None
    # prefer bold name patterns if your template uses **Name**
    bold = re.findall(r"\*\*([A-Za-z][A-Za-z0-9\-]{3,})\*\*", text)
    if bold:
        return bold[0]

    # otherwise, first capitalized word (avoid common sentence starters)
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9\-]{3,}\b", text)
    if not candidates:
        return None

    # filter out very common words that may appear capitalized
    blacklist = {"Please", "While", "Sedation", "For", "In", "If", "Do", "We", "Could", "This"}
    for c in candidates:
        if c not in blacklist:
            return c
    return None


def is_followup_about_previous_product(curr: str, conv_id: str) -> bool:
    """
    Detect follow-up questions that rely on previous product context,
    e.g., "what is its dosage", "what about side effects", "how to use it".
    """
    if not last_product_mentioned.get(conv_id):
        return False

    curr_n = _norm(curr)

    # pronoun-based followups
    pronoun_markers = {"its", "it", "this", "that", "this medicine", "this drug", "that medicine", "that drug"}
    if any(p in curr_n for p in pronoun_markers):
        # If user asked about common product-info fields, treat as follow-up
        follow_fields = ["dosage", "dose", "side effect", "side effects", "warning", "warnings",
                         "contraindication", "contraindications", "how to use", "usage", "precaution",
                         "precautions", "interaction", "interactions"]
        if any(f in curr_n for f in follow_fields):
            return True

    return False


def should_reset_context(prev_msg: str, curr_msg: str, conv_id: str) -> bool:
    """
    True => treat as NEW conversation (wipe memory)
    False => continue (keep memory)
    """
    curr = _norm(curr_msg)
    prev = _norm(prev_msg)

    if not prev:
        return False

    # 1) greetings/thanks -> treat as new conversation (matches your prompt rule)
    if curr in GREETINGS:
        return True

    # 2) explicit list/catalog requests -> reset scope (global list)
    for t in LIST_TRIGGERS:
        if t in curr:
            return True

    # 3) Generic follow-ups like "dosage" should CONTINUE (do not reset)
    if curr in FOLLOWUP_GENERIC:
        return False

    # 3.5) Pronoun-based followups like "what is its dosage" should CONTINUE
    if is_followup_about_previous_product(curr_msg, conv_id):
        return False

    # 4) Short messages reset ONLY if we have no active product context
    if len(curr.split()) <= 3 and not last_product_mentioned.get(conv_id):
        return True

    # 5) lexical overlap test: if almost no shared keywords, assume topic switch
    prev_tokens = _tokenize(prev)
    curr_tokens = _tokenize(curr)
    if not prev_tokens or not curr_tokens:
        return False

    overlap = len(prev_tokens & curr_tokens) / max(1, len(curr_tokens))

    # Tune this threshold if needed:
    # - more resets: 0.20
    # - fewer resets: 0.10
    return overlap < 0.15


# ----------------------------
# Chat engine factory
# ----------------------------

def get_chat_engine(conv_id: str):
    if not conv_id:
        conv_id = "default"

    if conv_id not in chat_memories:
        chat_memories[conv_id] = ChatMemoryBuffer.from_defaults(token_limit=1024)
        chat_engines[conv_id] = index.as_chat_engine(
            chat_mode="condense_question",  # type: ignore
            memory=chat_memories[conv_id],
            text_qa_template=text_qa_template,
            similarity_top_k=10,
            response_mode="compact",
            # Keep your exact arg name/version behavior (typo included if it's working in your env)
            condense_question_promp=condense_prompt,
        )
    return chat_engines[conv_id]


# ----------------------------
# Routes
# ----------------------------

@app.route("/")
def index_route():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat_route():
    user_msg = (request.form.get("msg") or "").strip()
    conv_id = (request.form.get("conversation_id") or "").strip()
    if not conv_id:
        conv_id = "default"

    if not user_msg:
        return jsonify({"response": "Please enter a question."})

    try:
        # Context switch: reset memory if user changed topic
        prev_msg = last_user_msgs.get(conv_id, "")
        if should_reset_context(prev_msg, user_msg, conv_id):
            chat_memories.pop(conv_id, None)
            chat_engines.pop(conv_id, None)
            # Do NOT clear last_product_mentioned here; it can help with immediate follow-ups
            # But if you want strict reset, uncomment:
            # last_product_mentioned.pop(conv_id, None)

        chat_engine = get_chat_engine(conv_id)
        response_obj = chat_engine.chat(user_msg)

        answer = (str(response_obj) if response_obj is not None else "").strip()
        if not answer:
            answer = "I could not generate a response."

        # Store last user message after successful processing
        last_user_msgs[conv_id] = user_msg

        # Try to capture last product mentioned from the assistant answer
        prod = extract_product_name_from_answer(answer)
        if prod:
            last_product_mentioned[conv_id] = prod

    except Exception as e:
        print(f"Error in /get handler: {e!r}")
        answer = "An error occurred while generating a response."

    return jsonify({"response": answer, "conversation_id": conv_id})


@app.route("/reset", methods=["POST"])
def reset_conversation():
    conv_id = (request.form.get("conversation_id") or "").strip()
    if not conv_id:
        conv_id = "default"

    chat_memories.pop(conv_id, None)
    chat_engines.pop(conv_id, None)
    last_user_msgs.pop(conv_id, None)
    last_product_mentioned.pop(conv_id, None)

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
