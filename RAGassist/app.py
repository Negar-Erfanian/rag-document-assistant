import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

DB_PATH = "vectorstore"

def load_rag_pipeline():

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    llm = ChatOpenAI(temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain


def main():

    qa = load_rag_pipeline()

    print("RAG Document Assistant")

    while True:

        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        result = qa.run(query)

        print("\nAnswer:", result)


if __name__ == "__main__":
    main()