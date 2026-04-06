import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "text_files")
DB_PATH = os.path.join(BASE_DIR, "vectorstore")

def ingest():
    dir_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",  ## Pattern to match files
        loader_cls=TextLoader,  ##loader class to use
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=False

    )

    documents = dir_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(DB_PATH)

    print("Vector database created.")

if __name__ == "__main__":
    ingest()