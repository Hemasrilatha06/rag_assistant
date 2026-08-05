import os
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ==========================================================
# 1. Initialize API Key
# ==========================================================

AGENTICAI_KEY = os.getenv("AgenticAI_KEY")


# ==========================================================
# 2. Initialize LLM & Embeddings
# ==========================================================

llm = ChatGoogleGenerativeAI(
    model="models/gemma-4-31b-it",
    google_api_key=AGENTICAI_KEY,
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=AGENTICAI_KEY
)


# ==========================================================
# 3. Load Documents
# ==========================================================

big_paragraph = """
The Internet is a global system of interconnected computer networks
that uses the Internet protocol suite (TCP/IP) to communicate between
networks and devices.

The origins of the Internet date back to the development of packet
switching and research commissioned by the United States Department
of Defense in the 1960s.

The primary precursor network was the ARPANET.

Today, the Internet supports cloud computing, video conferencing,
social media, file sharing, online education, and many other
applications.
"""

documents = [
    Document(page_content=big_paragraph)
]


# ==========================================================
# 4. Split Documents
# ==========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# ==========================================================
# 5. Create Vector Store
# ==========================================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# ==========================================================
# 6. Prompt Template
# ==========================================================

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI Assistant.

Answer only from the given context.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ==========================================================
# 7. Helper Function
# ==========================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==========================================================
# 8. Create RAG Chain
# ==========================================================

from operator import itemgetter

rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ==========================================================
# 9. FastAPI App
# ==========================================================

app = FastAPI(
    title="RAG LangServe API"
)


# ==========================================================
# 10. Request Model
# ==========================================================

class Query(BaseModel):
    question: str = Field(
        description="Ask a question to the RAG Assistant"
    )


# ==========================================================
# 11. REST Endpoint
# ==========================================================

@app.post("/ask")
def ask(query: Query):
    answer = rag_chain.invoke(query.question)
    return {
        "question": query.question,
        "answer": answer
    }


# ==========================================================
# 12. LangServe Endpoint
# ==========================================================

add_routes(
    app,
    rag_chain,
    path="/rag"
)


# ==========================================================
# 13. Root Endpoint
# ==========================================================

@app.get("/")
def root():
    return {
        "message": "RAG LangServe API is running"
    }


# ==========================================================
# 14. Run FastAPI
# ==========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
