from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain import PromptTemplate, LLMChain
from langchain_ollama import OllamaLLM
loader= PyPDFDirectoryLoader("pdfs")
 
data=loader.load()
data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=15)
text_chunk = text_splitter.split_documents(data)
emdedding= OllamaEmbeddings(model="snowflake-arctic-embed:latest " )
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=emdedding,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
#adding data into the database
uuids = [str(uuid4()) for _ in range(len(text_chunk))]
 
vector_store.add_documents(documents=text_chunk, ids=uuids)
#llm model loaded
llm=OllamaLLM(model="stablelm-zephyr:3b")
#templete for the llm model
template = """
You are an expert assistant specializing in Apple's history. Provide accurate, concise, and contextually relevant answers based solely on the information provided. Do not speculate or give opinions. Avoid referencing the context directly or acknowledging its existence. Simply respond to the user query with a direct and informative answer.

### Context:
{context}

### User Query:
{query}

### Your Response:

"""
prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=template,
)
apple_history_chain = LLMChain(llm=llm, prompt=prompt)
while True:
    user_query = input("Ask a question about Apple's history (type 'exit' to quit): ")
    
    if user_query.lower() == "exit":
        print("Goodbye!")
        break
    
    # Fetch relevant context using the context function
    results = vector_store.similarity_search(user_query, k=2)
    
    # Run the chain with the given context and query
    response = apple_history_chain.run(llm=llm, context=results, query=user_query)
    print("\nResponse:\n", response)



