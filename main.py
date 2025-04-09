from fastapi import FastAPI
from faiss_ai import router  # Import your router

app = FastAPI()

app.include_router(router, prefix="/faiss", tags=["FAISS"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG QA API!"}
