from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain

# 1. Load embeddings
embeddings = download_embeddings()  # your PubMedBERT embeddings

# 2. Connect to existing Pinecone index
index_name = "medicine-chatbot-trials1"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 3. Convert vector store into retriever
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}  # top-3 matches
)

# 4. Initialize your LLM
chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

# 5. Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medicine assistant. Answer the questions based on the provided documents."),
    ("human", "Context: {context}\nQuestion: {input}\nAnswer:")
])

# 6. Create the QA chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

# 7. Combine retriever and QA chain into a RAG chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 8. Example usage
query = "What are the side effects of ibuprofen?"
result = rag_chain.run(query)
print(result)
