from typing import List, Dict, Any
from model.core_interface import LLMInterface
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from utils.promptTemplate import RAG_PROMPT_TEMPLATE
from utils.app_config import (
    LLM_PROVIDER,
    LLM_MODELS,
    GOOGLE_API_KEY,
    OPENAI_API_KEY
)


def _format_context(context: List[Dict[str, Any]]) -> str:
    """
    Formats the retrieved context chunks to include the rich metadata,
    including the product name, so the LLM can use it in the answer.
    """
    context_chunks = []
    for doc in context:
        text = doc.get("text", "")
        # --- Retrieve product_name from metadata (ADDED) ---
        product = doc.get("product_name", "Unknown Product") 
        page = doc.get("page", 0)
        
        # Prepend the rich metadata header to the chunk
        # This header informs the LLM exactly where the information came from.
        chunk_header = f"--- Context (Product: {product}, Page: {page}) ---"
        context_chunks.append(f"{chunk_header}\n{text}")
    
    return "\n\n".join(context_chunks)

class GeminiLLM(LLMInterface):
    """Implementation for Google's Gemini model."""
    def __init__(self):
        self.model_name = LLM_MODELS["gemini"]
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.5,
                convert_system_message_to_human=True
            )
            print(f"Loaded GeminiLLM: {self.model_name}")
        except Exception as e:
            print(f"Error loading ChatGoogleGenerativeAI: {e}")
            raise
        self._setup_chain()

    def _setup_chain(self):
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_response(self, prompt: str, context: List[Dict[str, Any]]) -> str:
        # Uses the _format_context function above
        context_str = _format_context(context)
        return self.chain.invoke({"context": context_str, "question": prompt})

class OpenAILLM(LLMInterface):
    """Implementation for OpenAI's GPT models."""
    def __init__(self):
        self.model_name = LLM_MODELS["openai"]
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=OPENAI_API_KEY,
                temperature=0.5
            )
            print(f"Loaded OpenAILLM: {self.model_name}")
        except Exception as e:
            print(f"Error loading ChatOpenAI: {e}")
            raise
        self._setup_chain()

    def _setup_chain(self):
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_response(self, prompt: str, context: List[Dict[str, Any]]) -> str:
        # Uses the _format_context function above
        context_str = _format_context(context)
        return self.chain.invoke({"context": context_str, "question": prompt})

# --- Factory Function ---

def get_llm() -> LLMInterface:
    """
    Factory function to get the configured LLM
    based on the 'LLM_PROVIDER' in app_config.py.
    """
    if LLM_PROVIDER == "openai":
        return OpenAILLM()
    elif LLM_PROVIDER == "gemini":
        return GeminiLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")
