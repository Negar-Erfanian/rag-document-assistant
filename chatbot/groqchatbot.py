import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load environment variables
load_dotenv()

# Streamlit UI
st.title("AI Chatbot using LangChain + Groq")

user_input = st.text_input("Ask a question")

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "Question: {question}")
])

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

# Run
if user_input:
    response = chain.invoke({"question": user_input})
    st.write(response)