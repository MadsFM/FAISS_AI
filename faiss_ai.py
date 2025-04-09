import os
import psycopg2
from fastapi import APIRouter, Query
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

#Setup Database
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
)
cur = conn.cursor()
cur.execute("SELECT customer_name, issue_details, issue_summary FROM support_tickets")

# Build documents from DB
db_docs = [
    Document(
        page_content=f"Customer: {row[0]}\nSummary: {row[2]}\nIssue: {row[1]}",
        metadata={"source": "postgresql", "customer_name": row[0]}
    )
    for row in cur.fetchall()
]

# Alternative load .txt File
file_path = "files/history.txt"
split_docs = []

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().strip()
        if text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents([Document(page_content=text)])
            print(f"Loaded {len(split_docs)} chunks from history.txt")
        else:
            print("history.txt is empty. Skipping.")
else:
    print("No history.txt file found. Skipping.")

# Faiss Embedding
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

index_path = "faiss_index"
index_file = os.path.join(index_path, "index.faiss")

# Combine .txt and database
all_docs = db_docs + split_docs

if not os.path.exists(index_file):
    print("Creating new FAISS index from available documents...")
    vector_store = FAISS.from_documents(all_docs, embedding_model)
    vector_store.save_local(index_path)
else:
    print("Loading existing FAISS index...")
    vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

# Retrieval QA
retriever = vector_store.as_retriever()
llm = OllamaLLM(model="gemma2:9b")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Endpoint
@router.get("/ask")
def ask_question(query: str = Query(..., description="Your question")):
    try:
        response = qa_chain.run(query)
        return {"question": query, "answer": response}
    except Exception as e:
        return {"error": str(e)}