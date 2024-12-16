Apple History Assistant

This project demonstrates how to create an AI-powered assistant using LangChain, Chroma, and Ollama LLMs. The assistant specializes in answering questions about Apple's history by leveraging context from PDF documents.

Features

Load PDF documents and preprocess their content into manageable text chunks.

Embed text chunks using Ollama embeddings.

Store embeddings in a Chroma vector database for efficient similarity search.

Query a language model (LLM) with contextually relevant information to generate accurate answers.

Requirements

Dependencies

Install the required Python packages using pip:

pip install langchain chromadb

Additionally, ensure you have access to the Ollama models and APIs required for OllamaEmbeddings and OllamaLLM. You may need specific authentication credentials for Ollama.

How It Works

Code Workflow

Load PDF Documents:

The PyPDFDirectoryLoader loads all PDF files from the pdfs directory.

Split Documents into Chunks:

The RecursiveCharacterTextSplitter breaks the PDF text into smaller chunks for efficient processing and embedding.

Embed and Store in Chroma:

Each chunk is embedded using the OllamaEmbeddings model, and embeddings are stored in a Chroma vector store for similarity search.

Interactive Query:

Users can ask questions interactively. The system retrieves the most relevant text chunks using vector similarity search and uses the OllamaLLM to generate a response.

File Structure

.
|-- pdfs/               # Directory containing the PDF documents.
|-- chroma_langchain_db # Directory for the Chroma vector store database.
|-- main.py             # Main Python script.
|-- README.md           # Project README file (this file).

Usage

1. Add PDF Files

Place your PDF files in the pdfs/ directory. These files should contain information about Apple's history.

2. Run the Script

Run the Python script:

python main.py

3. Ask Questions

Interact with the assistant by typing questions related to Apple's history. For example:

Ask a question about Apple's history (type 'exit' to quit): When was Apple founded?
Response:
 Apple was founded on April 1, 1976.

Type exit to quit the interactive loop.

Key Code Highlights

PDF Loading

loader = PyPDFDirectoryLoader("pdfs")
data = loader.load()

Text Splitting

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=15)
text_chunks = text_splitter.split_documents(data)

Embedding and Chroma Vector Store

embedding = OllamaEmbeddings(model="snowflake-arctic-embed:latest")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)
vector_store.add_documents(documents=text_chunks)
vector_store.persist()

Query Loop

while True:
    user_query = input("Ask a question about Apple's history (type 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("Goodbye!")
        break

    results = vector_store.similarity_search(user_query, k=2)
    context = "\n".join([doc.page_content for doc in results])
    response = apple_history_chain.run(context=context, query=user_query)
    print("\nResponse:\n", response)

Future Improvements

Add a user interface (UI) for a more intuitive experience.

Enhance the assistant to handle multi-modal inputs (e.g., images or videos).

Incorporate advanced LLM models for improved accuracy.

License

This project is licensed under the MIT License. Feel free to use and adapt it for your own projects.

Acknowledgments

Special thanks to the developers of LangChain, Chroma, and Ollama for their powerful tools and APIs.

