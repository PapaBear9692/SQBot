from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from core.interfaces import LLMInterface
import config

class GeminiLLM(LLMInterface):
    """Implementation for Google's Gemini model."""
    def __init__(self, model_name: str = config.LLM_MODELS["gemini"]):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.3
            )
            print(f"Loaded GeminiLLM: {model_name}")
        except Exception as e:
            print(f"Error loading ChatGoogleGenerativeAI: {e}")
            print("Please ensure GOOGLE_API_KEY is set correctly in your .env file.")
            raise
        self._setup_chain()

    def _setup_chain(self):
        prompt_template = """
        You are a helpful medical assistant. Answer the user's question based *only* on the following context.
        If the context does not contain the answer, say "I am sorry, but the provided context does not contain that information."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_response(self, prompt: str, context: List[Document]) -> str:
        context_str = "\n\n".join([doc.page_content for doc in context])
        return self.chain.invoke({"context": context_str, "question": prompt})

class OpenAILLM(LLMInterface):
    """Implementation for OpenAI's GPT models."""
    def __init__(self, model_name: str = config.LLM_MODELS["openai"]):
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=config.OPENAI_API_KEY,
                temperature=0.3
            )
            print(f"Loaded OpenAILLM: {model_name}")
        except Exception as e:
            print(f"Error loading ChatOpenAI: {e}")
            print("Please ensure OPENAI_API_KEY is set correctly in your .env file.")
            raise
        self._setup_chain()

    def _setup_chain(self):
        # Using the same prompt template for consistency
        prompt_template = """
        You are a helpful medical assistant. Answer the user's question based *only* on the following context.
        If the context does not contain the answer, say "I am sorry, but the provided context does not contain that information."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        self.chain = prompt | self.llm | StrOutputParser()

    def generate_response(self, prompt: str, context: List[Document]) -> str:
        context_str = "\n\n".join([doc.page_content for doc in context])
        return self.chain.invoke({"context": context_str, "question": prompt})

# --- Factory Function ---

def get_llm() -> LLMInterface:
    """Factory function to get the configured LLM."""
    if config.LLM_PROVIDER == "openai":
        return OpenAILLM()
    elif config.LLM_PROVIDER == "gemini":
        return GeminiLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")
