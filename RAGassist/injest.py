from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
DB_PATH = "vectorstore"

def ingest_documents():

    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(DB_PATH)

    print("Vector database created")


if __name__ == "__main__":
    ingest_documents()