print("🚀 FastAPI starting...")

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    question: str

# Health check
@app.get("/")
def home():
    print("Home route working")
    return {"message": "API is running"}

# Lazy import (IMPORTANT FIX)
@app.post("/ask")
def ask(query: QueryRequest):
    from retr_and_gen import ask_question  # moved inside function
    return ask_question(query.question)