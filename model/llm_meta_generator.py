from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json


class LLMMetaGenerator:
    """
    Generates semantic metadata (topic, entities, summary) for text chunks using Gemini.
    """

    def __init__(self, model_name="gemini-2.5-flash"):
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            max_output_tokens=200,
        )

        # Define the metadata prompt
        self.prompt = PromptTemplate.from_template(
            """You are a metadata generator for a document embedding system.
Given the following text chunk, produce a short, well-structured JSON metadata containing:
- "topic": the main subject or domain of the text
- "entities": a list of key entities, terms, or important words
- "summary": a concise 1 sentence description of what this chunk is about

Text Chunk:
{text}

Return valid JSON only, no explanations.
"""
        )

    def generate_metadata(self, text: str) -> dict:
        """
        Generate metadata using Gemini for a given text chunk.
        """
        try:
            # Run LLM and format output
            response = self.llm.invoke(self.prompt.format(text=text))
            content = response.content if hasattr(response, "content") else str(response)

            # Try parsing LLM output as JSON
            meta = json.loads(content)
            if isinstance(meta, dict):
                return meta
            else:
                return {"summary": content}
        except Exception as e:
            print(f"[LLMMetaGenerator] Metadata generation failed: {e}")
            # Fallback: simple summary snippet
            return {"summary": text[:150]}
