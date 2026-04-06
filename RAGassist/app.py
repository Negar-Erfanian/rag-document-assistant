import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "vectorstore")

def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa


def main():

    qa = load_rag()

    print("\nGroq RAG Assistant")
    print("Type 'exit' to quit\n")

    while True:

        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        result = qa.run(query)

        print("\nAnswer:")
        print(result)
        print("\n")


if __name__ == "__main__":
    main()