from core.embedders import get_embedder
from core.llms import get_llm
from core.vector_stores import get_vector_store
import config

class RAGService:
    """
    Orchestrates the RAG (Retrieval-Augmented Generation) process.
    This is the core "Model" in our MVC-like structure.
    """
    def __init__(self):
        print("Initializing RAGService...")
        # Get the configured components from the factories
        self.embedder = get_embedder()
        self.llm = get_llm()
        self.vector_store = get_vector_store()
        print("RAGService initialized successfully.")

    def get_answer(self, prompt: str) -> str:
        """
        Retrieves context and generates an answer for a given prompt.
        """
        try:
            # 1. Embed the user's prompt (query)
            print(f"Embedding query: '{prompt}'")
            query_embedding = self.embedder.embed_query(prompt)
            
            # 2. Retrieve relevant documents (context)
            print(f"Querying vector store (top_k={config.TOP_K_RESULTS})...")
            context_docs = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=config.TOP_K_RESULTS
            )
            
            if not context_docs:
                print("No relevant context found.")
                return "I am sorry, but I could not find any relevant information to answer your question."

            print(f"Retrieved {len(context_docs)} context documents.")
            
            # 3. Call the LLM with the prompt and context
            print("Generating response from LLM...")
            answer = self.llm.generate_response(
                prompt=prompt,
                context=context_docs
            )
            
            print("Response generated.")
            return answer

        except Exception as e:
            print(f"Error in RAGService.get_answer: {e}")
            return "An error occurred while processing your request. Please try again."

# --- Singleton Instance ---
# Create a single instance to be shared by the Flask app
# This avoids re-initializing models on every request
try:
    rag_service_instance = RAGService()
except Exception as e:
    print(f"Failed to initialize RAGService: {e}")
    rag_service_instance = None
