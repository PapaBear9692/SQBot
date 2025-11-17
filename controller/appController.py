
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings


# 1. Load Pinecone index as retriever
index_name = "sqbot-index"
embedder = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedder 
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 2. Load Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_output_tokens=400
)

# 3. Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an medical expert assistant. Use ONLY the provided context."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 4. RAG pipeline function
def rag_pipeline(query):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm
    result = chain.invoke({"question": query, "context": context})

    return result.content

