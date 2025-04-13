from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_engine import RAGEngine

app = FastAPI()
rag = RAGEngine()

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(q: Question):
    answer = rag.ask(q.query)
    return {"question": q.query, "answer": answer}
