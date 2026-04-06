# RAG Document Assistant

This project demonstrates a Retrieval-Augmented Generation (RAG) system
for question answering over custom document collections.

Architecture

Documents → Chunking → Embeddings → Vector Database → Retriever → LLM

Tech Stack
- Python
- LangChain
- FAISS
- OpenAI API

Features
- Document ingestion and preprocessing
- Embedding generation
- Semantic retrieval
- LLM-powered answer generation

Usage

1. Install dependencies

pip install -r requirements.txt

2. Set OpenAI API key

export OPENAI_API_KEY=your_key

3. Create vector index

python ingest.py

4. Start the assistant

python app.py
