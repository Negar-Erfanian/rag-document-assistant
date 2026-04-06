from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.chains import PebbloRetrievalQA
import os

loader = PyPDFLoader("vehicle.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents,
    embeddings
)

retriever = vectorstore.as_retriever()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

qa_chain = PebbloRetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    # Standard RetrievalQA uses "stuff" by default,
    # but Pebblo often requires it explicitly
    chain_type="stuff"
)

print(qa_chain.invoke({"query": "Summarize the document"}))