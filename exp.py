import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings  # Replace if using a different embedding model
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings
import pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
import chromadb
from langchain.vectorstores import Chroma
 
loader= PyPDFDirectoryLoader("pdfs")
loader
 
data=loader.load()
data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunk = text_splitter.split_documents(data)
text_chunk
emdedding= OllamaEmbeddings(model="llama3.2:1b" )
from langchain_chroma import Chroma
 
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=emdedding,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
from uuid import uuid4
 
from langchain_core.documents import Document
uuids = [str(uuid4()) for _ in range(len(text_chunk))]
 
vector_store.add_documents(documents=text_chunk, ids=uuids)
results = vector_store.similarity_search(
    "WHO IS STEVE JOBS",
    k=2,
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
from langchain_ollama import OllamaLLM
llm=OllamaLLM(model="llama3.2:1b")
template = """
You are an expert assistant specializing in Apple's history. Your job is to answer questions accurately using the provided context. 
If the context doesn't contain enough information, ask the user to provide additional details. Never make up answers.
 
### Context:
{context}
 
### User Query:
{query}
 
### Your Response:
"""
from langchain import PromptTemplate, LLMChain
# Step 2: Create a PromptTemplate
prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=template,
)
apple_history_chain = LLMChain(llm=llm, prompt=prompt)
query = "When was Apple founded, and who were the founders?"
response = apple_history_chain.run(context=results, query=query)
print(response)
 