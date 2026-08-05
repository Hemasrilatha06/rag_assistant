import os
import uvicorn
from operator import itemgetter

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="RAG LangServe API",
    version="1.0.0"
)


# ==========================================================
# Request Model
# ==========================================================

class Query(BaseModel):
    question: str = Field(
        description="Ask a question to the RAG Assistant"
    )


rag_chain = None


# ==========================================================
# Startup
# ==========================================================

@app.on_event("startup")
def startup():

    global rag_chain

    api_key = os.getenv("AgenticAI_KEY")

    if not api_key:
        raise ValueError(
            "AgenticAI_KEY environment variable is missing."
        )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="models/gemma-4-31b-it",
        google_api_key=api_key,
        temperature=0,
    )

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # ======================================================
    # Load Existing FAISS Index
    # ======================================================

    vector_store = FAISS.load_local(
        ".",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    # ======================================================
    # Prompt
    # ======================================================

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Answer ONLY from the given context.

If the answer is not present in the context,
reply exactly:

I don't know.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # ======================================================
    # Helper
    # ======================================================

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ======================================================
    # RAG Chain
    # ======================================================

    rag_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | format_docs,

            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    ).with_types(input_type=Query)

    # ======================================================
    # LangServe Route
    # ======================================================

    add_routes(
        app,
        rag_chain,
        path="/rag",
    )


# ==========================================================
# REST API
# ==========================================================

@app.post("/ask")
def ask(query: Query):

    return {
        "question": query.question,
        "answer": rag_chain.invoke(
            {
                "question": query.question
            }
        ),
    }


# ==========================================================
# Root
# ==========================================================

@app.get("/")
def root():

    return {
        "message": "RAG LangServe API is running"
    }


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
