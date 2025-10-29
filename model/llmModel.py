from typing import List, Dict, Any
from model.core_interface import LLMInterface
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from utils.promptTemplate import RAG_PROMPT_TEMPLATE

from utils.app_config import (
    LLM_PROVIDER,
    LLM_MODELS,
    GOOGLE_API_KEY,
    OPENAI_API_KEY
)



class GeminiLLM(LLMInterface):
    """Implementation for Google's Gemini model."""
    def __init__(self):
        self.model_name = LLM_MODELS["gemini"]
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.4,
                convert_system_message_to_human=True  # For compatibility
            )
            print(f"Loaded GeminiLLM: {self.model_name}")
        except Exception as e:
            print(f"Error loading ChatGoogleGenerativeAI: {e}")
            raise
        self._setup_chain()

    def _setup_chain(self):
        """Sets up the LangChain chain for RAG."""
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_response(self, prompt: str, context: List[Dict[str, Any]]) -> str:
        context_str = "\n\n".join([doc.get("text", "") for doc in context])
        return self.chain.invoke({"context": context_str, "question": prompt})


class OpenAILLM(LLMInterface):
    """Implementation for OpenAI's GPT models."""
    def __init__(self):
        self.model_name = LLM_MODELS["openai"]
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key=OPENAI_API_KEY,
                temperature=0.3
            )
            print(f"Loaded OpenAILLM: {self.model_name}")
        except Exception as e:
            print(f"Error loading ChatOpenAI: {e}")
            raise
        self._setup_chain()

    def _setup_chain(self):
        """Sets up the LangChain chain for RAG."""
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_response(self, prompt: str, context: List[Dict[str, Any]]) -> str:
        context_str = "\n\n".join([doc.get("text", "") for doc in context])
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
