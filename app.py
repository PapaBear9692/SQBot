import os
import secrets
import logging
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate

from embedder import init_settings_and_storage
from prompt import PROMPT_TEMPLATE

# ---------- Load environment ----------
load_dotenv()

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- LlamaIndex setup ----------
text_qa_template = PromptTemplate(PROMPT_TEMPLATE)

logger.info("Initializing LlamaIndex + Pinecone via embedder.init_settings_and_storage()...")
storage_context = init_settings_and_storage()
vector_store = storage_context.vector_store

index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(
    similarity_top_k=7,
    response_mode="compact",
    text_qa_template=text_qa_template,
)

# ---------- FastAPI app ----------
app = FastAPI()
templates = Jinja2Templates(directory="templates")  # same folder Flask uses

# NEW: static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

def run_query(user_msg: str):
    """Same query logic as Flask version."""
    try:
        response_obj = query_engine.query(user_msg)
        answer = (str(response_obj) if response_obj is not None else "").strip()
        if not answer:
            answer = "I could not generate a response."

        sources = []
        for sn in getattr(response_obj, "source_nodes", []) or []:
            try:
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
                logger.warning(f"Error formatting source node: {e!r}")
    except Exception as e:
        logger.exception(f"Query engine error: {e!r}")
        answer = "An error occurred while generating a response."
        sources = []

    return answer, sources


# ---------- UI route ----------
@app.get("/", response_class=HTMLResponse)
async def index_route(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# ---------- Legacy form endpoint (/get) ----------
@app.post("/get")
async def chat_route(msg: str = Form(None)):
    if not msg or not msg.strip():
        return JSONResponse({"response": "Please enter a question.", "sources": []})

    answer, sources = run_query(msg.strip())
    return JSONResponse({"response": answer, "sources": sources})


# ---------- Improved JSON API (/api/chat) ----------
@app.post("/api/chat")
async def chat_api(data: dict):
    user_msg = (data.get("msg") or "").strip()

    if not user_msg:
        return JSONResponse(
            {
                "ok": False,
                "error": {
                    "type": "validation_error",
                    "message": "Field 'msg' is required and cannot be empty.",
                },
            },
            status_code=400,
        )

    try:
        answer, sources = run_query(user_msg)
        return JSONResponse(
            {
                "ok": True,
                "answer": answer,
                "sources": sources,
            }
        )
    except Exception as e:
        logger.exception(f"Unexpected error: {e!r}")
        return JSONResponse(
            {
                "ok": False,
                "error": {
                    "type": "server_error",
                    "message": "An internal error occurred while generating a response.",
                },
            },
            status_code=500,
        )
