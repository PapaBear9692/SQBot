from typing import Tuple, List, Dict, Any
from model.embedderModel import get_embedder
from model.llmModel import get_llm
from model.storageModel import PineconeStorage
from model.core_interface import EmbeddingInterface, LLMInterface, VectorStoreInterface
from utils.app_config import TOP_K_RESULTS

class RAGModel:
    """
    Orchestrates the RAG (Retrieval-Augmented Generation) process.
    This is the core "Model" in our MVC-like structure.
    """
    def __init__(self):
        print("Initializing RAGModel...")
        # Get the configured components from the factories
        self.embedder: EmbeddingInterface = get_embedder()
        self.llm: LLMInterface = get_llm()
        self.vector_store: VectorStoreInterface = PineconeStorage()
        
        # Connect to the Pinecone index immediately
        # This assumes the index is already created by the dataController
        self.vector_store.connect_to_index()
        print("RAGModel initialized successfully.")

    def generate_answer(self, prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves context and generates an answer for a given prompt.
        
        Returns:
            - The generated answer (str).
            - The retrieved source documents (List[Dict]).
        """
        try:
            # 1. Embed the user's prompt (query)
            print(f"Embedding query: '{prompt}'")
            query_embedding = self.embedder.embed_query(prompt)
            
            # 2. Retrieve relevant documents (context)
            print(f"Querying vector store (top_k={TOP_K_RESULTS})...")
            context_docs = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=TOP_K_RESULTS
            )
            
            if not context_docs:
                print("No relevant context found.")
                return "I am sorry, but I could not find any relevant information to answer your question.", []

            print(f"Retrieved {len(context_docs)} context documents.")
            
            # 3. Call the LLM with the prompt and context
            # The RAG_PROMPT_TEMPLATE is applied inside the LLM model
            print("Generating response from LLM...")
            answer = self.llm.generate_response(
                prompt=prompt,
                context=context_docs
            )
            
            print("Response generated.")
            return answer, context_docs

        except Exception as e:
            print(f"Error in RAGModel.generate_answer: {e}")
            return "An error occurred while processing your request. Please try again.", []