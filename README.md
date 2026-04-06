### Groq RAG Document Assistant

A lightweight Retrieval-Augmented Generation (RAG) system for answering questions over custom documents using semantic search and a large language model.

This project demonstrates a simple end-to-end AI pipeline including document ingestion, vector search, and LLM-based response generation.

### Overview

Retrieval-Augmented Generation improves language model responses by retrieving relevant information from external documents before generating an answer.

### Pipeline architecture:

Documents\
   ↓\
Chunking\
   ↓\
Embeddings\
   ↓\
Vector Database (FAISS)\
   ↓\
Retriever\
   ↓\
LLM (Groq / Llama)\
   ↓\
Answer

 Instead of relying only on model knowledge, the system retrieves document chunks from a vector database and uses them as context for the language model.

### Tech Stack
- Python
- LangChain
- FAISS vector database
- Groq API
- Llama 3.1 model
- Sentence Transformers embeddings 

### Project Structure
```
LangChain
│
├── data
│   └── text_files
│       └── *.txt        # Documents used for retrieval
│
├── RAGassist
│   ├── app.py           # RAG query application
│   ├── ingest.py        # Document ingestion pipeline
│   └── vectorstore      # Generated FAISS index
│
├── .env                 # API keys (not committed)
├── requirements.txt
└── README.md
```

## How It Works
### 1. Document Ingestion

ingest.py
loads documents from: 
```bash
data/text_files/
```

The script:

1. Loads all ```.txt``` files
2. Splits documents into chunks
3. Generates embeddings
4. Stores them in a FAISS vector database

Example parameters:

```
chunk_size = 500
chunk_overlap = 50
```
### 2. Vector Search

Embeddings are stored in a FAISS vector database, enabling fast semantic similarity search.

The index is saved to:
```
RAGassist/vectorstore
```
### 3. Retrieval + Generation

When a user asks a question:

1. The query is converted to an embedding
2. The system retrieves the top 3 relevant document chunks
3. The retrieved context is sent to the LLM
4. The LLM generates a grounded answer

The system uses:
```
llama-3.1-8b-instant
```
via the Groq API.

### Setup
1. Install dependencies
```
pip install -r requirements.txt
```
2. Add Groq API key


Create a ```.env``` file in the project root:
```
GROQ_API_KEY=your_api_key_here
```
3. Add documents

Place ```.txt``` files in:
```
data/text_files/
```
4. Create the vector database
```
python RAGassist/ingest.py
```
This will generate:
```
RAGassist/vectorstore/
```
5. Run the assistant
```
python RAGassist/app.py
```
Example:
```
Groq RAG Assistant

Ask a question: What is RAG?
```