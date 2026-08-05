import os
import faiss
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable.")

llm = ChatGoogleGenerativeAI(
    model="models/gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
)

text = """The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices."""
documents=[Document(page_content=text)]

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=splitter.split_documents(documents)

embeddings=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)
dim=len(embeddings.embed_query("hello"))
index=faiss.IndexFlatL2(dim)
vector_store=FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
vector_store.add_documents(chunks)
retriever=vector_store.as_retriever(search_kwargs={"k":2})

prompt=ChatPromptTemplate.from_template("""Use only the context.

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

if __name__=="__main__":
    while True:
        q=input("Ask: ")
        if q.lower() in {"exit","quit"}:
            break
        print(rag_chain.invoke(q))
