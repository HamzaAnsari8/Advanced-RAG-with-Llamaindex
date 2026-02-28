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

@app.post("/ask")
def ask(query: QueryRequest):
    print("Request received:", query.question)

    try:
        from retr_and_gen import ask_question
        response = ask_question(query.question)
        print("RAG response:", response)
        return response

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}