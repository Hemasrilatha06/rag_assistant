from fastapi import FastAPI
from pydantic import BaseModel
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langserve import add_routes

AGENTICAI_KEY = os.getenv("AgenticAI_KEY")

llm = ChatGoogleGenerativeAI(
    model="models/gemma-4-31b-it",
    google_api_key=AGENTICAI_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=AGENTICAI_KEY,
)

big_paragraph = """The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies.

The origins of the Internet date back to the development of packet switching and research commissioned by the United States Department of Defense in the 1960s. The primary precursor network was the ARPANET.

Today, the Internet supports cloud computing, video conferencing, social media, file sharing, and many other applications."""
documents=[Document(page_content=big_paragraph)]
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=splitter.split_documents(documents)
vector_store=FAISS.from_documents(chunks,embeddings)
retriever=vector_store.as_retriever(search_kwargs={"k":2})

prompt=ChatPromptTemplate.from_template(
"""You are a helpful assistant.
Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain=(
    {"context":retriever|format_docs,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app=FastAPI(title="RAG LangServe API")

class Query(BaseModel):
    question:str

@app.post("/ask")
def ask(query:Query):
    return {"answer":rag_chain.invoke(query.question)}

add_routes(app,rag_chain,path="/rag")

@app.get("/")
def root():
    return {"message":"LangServe API is running"}
