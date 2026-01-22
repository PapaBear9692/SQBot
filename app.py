import os
import secrets
import json
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI

from app_config import init_settings_and_storage
from prompt import PROMPT_TEMPLATE, ROUTER_PROMPT  # <-- add ROUTER_PROMPT in prompt.py


load_dotenv()

# ----------------------------
# App + LlamaIndex setup
# ----------------------------

text_qa_template = PromptTemplate(PROMPT_TEMPLATE)

app = Flask(__name__, static_url_path="/chat/static")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)

storage_context = init_settings_and_storage()
vector_store = storage_context.vector_store
index = VectorStoreIndex.from_vector_store(vector_store)

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)

# Router LLM (LLM call #1)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "models/gemini-2.5-flash-lite")
router_llm = GoogleGenAI(model=ROUTER_MODEL, api_key=GOOGLE_API_KEY, temperature=0.0)

# In-memory conversation state (demo)
last_user_msgs: Dict[str, str] = {}
last_product_mentioned: Dict[str, str] = {}


# ----------------------------
# Router helpers
# ----------------------------

def _render_router_prompt(template: str, variables: Dict[str, str]) -> str:
    """
    DO NOT use .format() because templates often contain JSON braces {}.
    We only replace known placeholders.
    """
    out = template
    for k, v in variables.items():
        out = out.replace("{" + k + "}", v or "")
    return out


def _extract_first_json_object(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()

    # Strip code fences if model adds them
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Router did not return JSON:\n{raw}")

    return json.loads(raw[start:end + 1])


def route_message(user_msg: str, conv_id: str) -> Dict[str, Any]:
    prompt = _render_router_prompt(
        ROUTER_PROMPT,
        {
            "last_product": last_product_mentioned.get(conv_id, ""),
            "last_user_message": last_user_msgs.get(conv_id, ""),
            "user_message": user_msg,
        },
    )

    raw = router_llm.complete(prompt).text
    data = _extract_first_json_object(raw)

    # Defaults for minimal schema
    data.setdefault("intent", "OTHER")
    data.setdefault("ignore_history", False)
    data.setdefault("followup", False)
    data.setdefault("product_name", None)
    data.setdefault("retrieval_query", "")
    data.setdefault("needs_clarification", False)
    data.setdefault("clarification_question", "")

    # Normalize product_name (force string or None)
    pn = data.get("product_name")
    if isinstance(pn, str) and not pn.strip():
        data["product_name"] = None
    if not (pn is None or isinstance(pn, str)):
        data["product_name"] = None

    return data


def apply_followup_rule(route: Dict[str, Any], conv_id: str) -> Dict[str, Any]:
    """
    HARD RULE (ChatGPT-style):
    If followup=True and router didn't set product_name,
    inherit last known product_name (if available).
    """
    if route.get("followup") is True and not route.get("product_name"):
        lp = last_product_mentioned.get(conv_id)
        if lp:
            route["product_name"] = lp
            route["needs_clarification"] = False
            route["clarification_question"] = ""

    # Clarify ONLY if PRODUCT_INFO but still no product
    if route.get("intent") == "PRODUCT_INFO" and not route.get("product_name"):
        route["needs_clarification"] = True
        if not route.get("clarification_question"):
            route["clarification_question"] = "Which medicine/product are you asking about?"

    return route


# ----------------------------
# LlamaIndex-only retrieval helpers (NO metadata filtering)
# ----------------------------

def _nodes_to_context_str(nodes_with_scores) -> str:
    parts = []
    for i, nws in enumerate(nodes_with_scores, start=1):
        # no metadata usage for retrieval decisions
        text = nws.node.get_content(metadata_mode="none") if hasattr(nws.node, "get_content") else str(nws.node)
        text = (text or "").strip()
        if text:
            parts.append(f"[{i}]\n{text}")
    return "\n\n".join(parts).strip()


def retrieve_context(query: str, top_k: int = 10) -> str:
    local = index.as_retriever(similarity_top_k=top_k)
    nodes = local.retrieve(query)
    return _nodes_to_context_str(nodes)


def retrieve_product_list_context() -> str:
    """
    LlamaIndex-only. No metadata filters. No Pinecone fetch by id.

    We "pin" the prime nodes by asking for deterministic, unique queries that
    only those 'All Product List' nodes should match strongly.
    """
    # These should match content inside your Prime nodes.
    # Adjust the strings to whatever is most unique in those PDFs/nodes.
    q_pharma = "All Product List (Pharma) Square Pharmaceuticals PLC"
    q_agrovet = "All Product List (Agrovet) Square Pharmaceuticals PLC"
    q_herbal = "All Product List (Herbal) Square Pharmaceuticals PLC"

    # Pull 1 best node per category (cheap + stable)
    pharma_ctx = retrieve_context(q_pharma, top_k=1)
    herbal_ctx = retrieve_context(q_herbal, top_k=1)

    # You previously said ignore agrovet in list output (your prompt rules),
    # but you also said you have the node. We'll still retrieve it in case
    # you later want it. If you truly never want it, comment it out.
    agrovet_ctx = retrieve_context(q_agrovet, top_k=1)

    # If you want to strictly ignore agrovet list, remove agrovet_ctx below.
    combined = "\n\n".join([pharma_ctx, herbal_ctx, agrovet_ctx]).strip()
    return combined


# ----------------------------
# Generation (LLM call #2)
# ----------------------------

def generate_answer(user_msg: str, context_str: str) -> str:
    prompt = text_qa_template.format(context_str=context_str, query_str=user_msg)
    resp = Settings.llm.complete(prompt)
    return (resp.text or "").strip()


# ----------------------------
# Core handler (keeps /get short)
# ----------------------------

def handle_chat_message(user_msg: str, conv_id: str) -> Tuple[str, str]:
    # Router (LLM #1)
    route = route_message(user_msg, conv_id)
    route = apply_followup_rule(route, conv_id)

    # Save last user message
    last_user_msgs[conv_id] = user_msg

    if route.get("needs_clarification"):
        return (route.get("clarification_question") or "Could you clarify your question?", conv_id)

    intent = route.get("intent", "OTHER")
    ignore_history = route.get("ignore_history", False)
    product_name = route.get("product_name")
    retrieval_query = (route.get("retrieval_query") or "").strip()

    # PRODUCT_LIST: deterministic LlamaIndex retrieval (no metadata)
    if intent == "PRODUCT_LIST":
        context_str = retrieve_product_list_context()
        answer = generate_answer(user_msg, context_str) or "I could not generate a response."
        return (answer, conv_id)

    # Build retrieval query for product info
    if intent == "PRODUCT_INFO":
        if not retrieval_query:
            retrieval_query = product_name or user_msg
        elif product_name and product_name.lower() not in retrieval_query.lower():
            retrieval_query = f"{product_name} {retrieval_query}".strip()

    elif not retrieval_query:
        retrieval_query = user_msg

    # Retrieval (LlamaIndex only)
    # ignore_history affects only how router/generator behave, not vector retrieval filtering.
    context_str = retrieve_context(retrieval_query, top_k=RETRIEVAL_TOP_K)

    # Generation (LLM #2)
    answer = generate_answer(user_msg, context_str) or "I could not generate a response."

    # Update last product only when we are confident (router decided)
    if product_name:
        last_product_mentioned[conv_id] = product_name

    return (answer, conv_id)


# ----------------------------
# Routes
# ----------------------------

@app.route("/")
def index_route():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat_route():
    user_msg = (request.form.get("msg") or "").strip()
    conv_id = (request.form.get("conversation_id") or "").strip() or "default"

    if not user_msg:
        return jsonify({"response": "Please enter a question.", "conversation_id": conv_id})

    try:
        answer, cid = handle_chat_message(user_msg, conv_id)
    except Exception as e:
        print(f"Error in /get handler: {e!r}")
        answer, cid = "An error occurred while generating a response.", conv_id

    return jsonify({"response": answer, "conversation_id": cid})


@app.route("/reset", methods=["POST"])
def reset_conversation():
    conv_id = (request.form.get("conversation_id") or "").strip() or "default"
    last_user_msgs.pop(conv_id, None)
    last_product_mentioned.pop(conv_id, None)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False)
