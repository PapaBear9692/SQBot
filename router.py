# router.py
import os
import json
from typing import Any, Dict, Tuple, Generator
from collections import defaultdict, deque
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.postprocessor import SentenceTransformerRerank


from app_config import CACHE_DIR
from prompt import PROMPT_TEMPLATE, ROUTER_PROMPT


# =====================================================================
# Shared in-memory state (per conv_id)
# =====================================================================
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "3"))
chat_history = defaultdict(lambda: deque(maxlen=CHAT_HISTORY_MAX_TURNS))
last_user_msgs: Dict[str, str] = {}

# LlamaIndex chat engine + memory per conversation
chat_memories: Dict[str, ChatMemoryBuffer] = {}
chat_engines: Dict[str, Any] = {}

CHAT_MEMORY_TOKENS = int(os.getenv("CHAT_MEMORY_TOKENS", "300"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
# Rerank settings
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"

# Pull this many candidates from Pinecone first (should be > RERANK_TOP_N)
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", str(max(RETRIEVAL_TOP_K, 30))))

# Keep this many after rerank (this is your "final top_k" effectively)
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", str(RETRIEVAL_TOP_K)))

# Good default cross-encoder reranker model
RERANK_MODEL = os.getenv("RERANK_MODEL", "NeuML/biomedbert-base-reranker")



# =====================================================================
# Phase 1: Google GenAI Router (intent, continuity, context switch, query)
# =====================================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "models/gemini-2.0-flash-lite")
router_llm = GoogleGenAI(model=ROUTER_MODEL, api_key=GOOGLE_API_KEY, temperature=0.0)


def _history_to_text(conv_id: str) -> str:
    turns = chat_history.get(conv_id, [])
    out = []
    for t in turns:
        role = (t.get("role") or "").upper()
        content = (t.get("content") or "").strip()
        if role and content:
            out.append(f"{role}: {content}")
    return "\n".join(out).strip()


def _render_router_prompt(template: str, variables: Dict[str, str]) -> str:
    """
    Avoid .format() because prompts contain JSON braces {}.
    Replace only known placeholders.

    If prompt.py doesn't yet include {chat_history}, we append it automatically.
    """
    out = (template or "").strip()
    if "{chat_history}" not in out and variables.get("chat_history"):
        out = out + "\n\nCHAT HISTORY:\n{chat_history}\n"

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

    return json.loads(raw[start : end + 1])


def _normalize_router_output(data: Dict[str, Any]) -> Dict[str, Any]:
    data = data or {}

    # Defaults
    data.setdefault("intent", "OTHER")
    data.setdefault("ignore_history", False)
    data.setdefault("followup", False)
    data.setdefault("product_name", None)
    data.setdefault("retrieval_query", "")
    data.setdefault("needs_clarification", False)
    data.setdefault("clarification_question", "")

    # Normalize types
    if not isinstance(data.get("intent"), str):
        data["intent"] = "OTHER"

    for k in ["ignore_history", "followup", "needs_clarification"]:
        if not isinstance(data.get(k), bool):
            data[k] = bool(data.get(k))

    for k in ["retrieval_query", "clarification_question"]:
        if not isinstance(data.get(k), str):
            data[k] = ""

    pn = data.get("product_name")
    if isinstance(pn, str) and not pn.strip():
        data["product_name"] = None
    if not (data.get("product_name") is None or isinstance(data.get("product_name"), str)):
        data["product_name"] = None

    print(f"Router output normalized: {data}")
    return data


def route_message(user_msg: str, conv_id: str) -> Dict[str, Any]:
    """
    Router call (LLM #1)
    - uses chat_history text + last_user_message + user_message
    - returns JSON schema from ROUTER_PROMPT
    """
    prompt = _render_router_prompt(
        ROUTER_PROMPT,
        {
            "chat_history": _history_to_text(conv_id),
            "last_user_message": last_user_msgs.get(conv_id, ""),
            "user_message": user_msg,
        },
    )

    raw = (router_llm.complete(prompt).text or "").strip()
    data = _extract_first_json_object(raw)
    data = _normalize_router_output(data)

    # Safety rule: if PRODUCT_INFO but no product_name, force clarification
    if data.get("intent") == "PRODUCT_INFO" and not data.get("product_name"):
        data["needs_clarification"] = True
        if not data.get("clarification_question"): #rerun router to try get product_name
            raw = (router_llm.complete(prompt).text or "").strip()
            data = _extract_first_json_object(raw)
            data = _normalize_router_output(data)
            if data.get("intent") == "PRODUCT_INFO" and not data.get("product_name"):
                data["needs_clarification"] = True
                if not data.get("clarification_question"):
                    data["clarification_question"] = "Sorry, I did not get it. Which medicine/product are you asking about?"

    print("Router Model Response Generated.")
    return data


def reset_conversation(conv_id: str) -> None:
    """
    Clears all state for a conversation id.
    Used when router says ignore_history / context switch.
    """
    last_user_msgs.pop(conv_id, None)
    chat_history.pop(conv_id, None)
    chat_memories.pop(conv_id, None)
    chat_engines.pop(conv_id, None)
    print(f"Conversation {conv_id!r} has been reset.")


# =====================================================================
# Phase 2: LlamaIndex Retrieval + Answer Generation
# =====================================================================
text_qa_template = PromptTemplate(PROMPT_TEMPLATE)


# def _get_or_create_chat_engine(index: VectorStoreIndex, conv_id: str) -> Any:
#     """
#     LlamaIndex chat engine: retrieval + memory
#     """
#     if conv_id not in chat_memories:
#         chat_memories[conv_id] = ChatMemoryBuffer.from_defaults(token_limit=CHAT_MEMORY_TOKENS)
#         chat_engines[conv_id] = index.as_chat_engine(
#             chat_mode="context",  # type: ignore
#             memory=chat_memories[conv_id],
#             similarity_top_k=RETRIEVAL_TOP_K,
#         )
#     return chat_engines[conv_id]

def _get_or_create_chat_engine(index: VectorStoreIndex, conv_id: str) -> Any:
    """
    LlamaIndex chat engine: retrieval + memory + (optional) rerank
    """
    if conv_id not in chat_memories:
        chat_memories[conv_id] = ChatMemoryBuffer.from_defaults(token_limit=CHAT_MEMORY_TOKENS)

        node_postprocessors = []
        similarity_top_k = RETRIEVAL_TOP_K

        if RERANK_ENABLED:
            # Ensure we fetch more candidates than we keep
            candidates = max(RERANK_CANDIDATES, RERANK_TOP_N)
            similarity_top_k = candidates

            rerank = SentenceTransformerRerank(
                model=RERANK_MODEL,
                top_n=RERANK_TOP_N,
            )
            node_postprocessors = [rerank]

        chat_engines[conv_id] = index.as_chat_engine(
            chat_mode="context",  # type: ignore
            memory=chat_memories[conv_id],
            similarity_top_k=similarity_top_k,          # Pinecone candidates
            node_postprocessors=node_postprocessors,    # local reranker
        )

    return chat_engines[conv_id]



def _nodes_to_context_str(nodes_with_scores) -> str:
    parts = []
    for i, nws in enumerate(nodes_with_scores, start=1):
        try:
            text = nws.node.get_content(metadata_mode="none")
        except Exception:
            text = str(getattr(nws, "node", nws))
        text = (text or "").strip()
        if text:
            parts.append(f"[{i}]\n{text}")
    return "\n\n".join(parts).strip()


def retrieve_context(index: VectorStoreIndex, query: str, top_k: int) -> str:
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return _nodes_to_context_str(nodes)


def retrieve_product_list_context(index: VectorStoreIndex) -> str:
    """
    Prime-node style retrieval via deterministic queries (no metadata assumptions).
    Adjust strings if needed to uniquely hit your product-list nodes.
    """
    q_pharma = "All Product List Pharma, Herbal, Prime_Node_Pharma, Prime_Node_Herbal"

    pharma_ctx = retrieve_context(index, q_pharma, top_k=3)

    print(f"Product list contexts retrieved: pharma_len={len(pharma_ctx)}")
    return "\n\n" + pharma_ctx.strip()


def generate_answer_from_context(user_msg: str, context_str: str) -> str:
    """
    Deterministic QA prompt style generation (no chat memory)
    """
    prompt = text_qa_template.format(context_str=context_str, query_str=user_msg)
    resp = Settings.llm.complete(prompt)
    return (resp.text or "").strip()


def generate_answer_chat(index: VectorStoreIndex, user_msg: str, expanded_query: str, conv_id: str) -> str:
    """
    Chat-engine generation (retrieval + memory) - non-stream
    """
    engine = _get_or_create_chat_engine(index, conv_id)
    message = f"""User question:
{user_msg}

Retrieval query:
{expanded_query}
"""
    resp = engine.chat(message)
    print("LLamaIndex chat engine response generated.")
    return (str(resp) or "").strip()


# ---------------------------------------------------------------------
# STREAMING: Phase 2 streaming generator (tokens)
# ---------------------------------------------------------------------
def generate_answer_chat_stream(
    index: VectorStoreIndex, user_msg: str, expanded_query: str, conv_id: str
) -> Generator[str, None, str]:
    """
    Chat-engine generation (retrieval + memory) - streaming.
    Yields tokens, and returns the full text at the end (StopIteration.value).
    """
    engine = _get_or_create_chat_engine(index, conv_id)
    message = f"""User question:
{user_msg}

Retrieval query:
{expanded_query}
"""
    resp = engine.stream_chat(message)

    chunks = []
    for tok in resp.response_gen:
        if tok:
            chunks.append(tok)
            yield tok

    return "".join(chunks).strip()


# =====================================================================
# Orchestration: phase1 -> program flow -> phase2
# =====================================================================
def handle_chat_message(index: VectorStoreIndex, user_msg: str, conv_id: str) -> Tuple[str, str]:
    """
    (UNCHANGED) Non-streaming base handler.
    """
    user_msg = (user_msg or "").strip()
    conv_id = (conv_id or "").strip() or "default"

    if not user_msg:
        return ("Please enter a question.", conv_id)

    # -------------------------
    # Phase 1: Router
    # -------------------------
    route = route_message(user_msg, conv_id)

    # Save last user message (router input for next turn)
    last_user_msgs[conv_id] = user_msg

    # Context switch: hard reset when router says ignore history
    if route.get("ignore_history") is True:
        reset_conversation(conv_id)

    # Clarification path
    if route.get("needs_clarification"):
        q = route.get("clarification_question") or "Could you clarify your question?"
        chat_history[conv_id].append({"role": "user", "content": user_msg})
        return (q, conv_id)

    intent = route.get("intent", "OTHER")
    expanded_query = (route.get("retrieval_query") or "").strip()
    product_name = route.get("product_name")

    # Build expanded query safely
    if intent == "PRODUCT_INFO":
        if not expanded_query:
            expanded_query = product_name or user_msg
        elif product_name and product_name.lower() not in expanded_query.lower():
            expanded_query = f"{product_name} {expanded_query}".strip()

    if not expanded_query:
        expanded_query = user_msg

    # -------------------------
    # Phase 2: LlamaIndex
    # -------------------------
    if intent == "PRODUCT_LIST" or intent == "SMALLTALK":
        ctx = retrieve_product_list_context(index)
        if not ctx:
            answer = "I couldn't find the product list in my data right now."
        else:
            answer = generate_answer_from_context(user_msg, ctx) or "I could not generate a response."
    # intent == "PRODUCT_INFO" or others
    else:
        answer = generate_answer_chat(index, user_msg, expanded_query, conv_id) or "I could not generate a response."

    # Store turns for router continuity
    chat_history[conv_id].append({"role": "user", "content": user_msg})
    chat_history[conv_id].append({"role": "assistant", "content": answer})

    return (answer, conv_id)

