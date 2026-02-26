from fastapi import FastAPI
from pydantic import BaseModel
from retr_and_gen import ask_question   # use your function

app = FastAPI()

# Request format
class QueryRequest(BaseModel):
    question: str


# Test route
@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}


# Main RAG route
@app.post("/ask")
def ask_question_api(request: QueryRequest):
    try:
        response = ask_question(request.question)   # ✅ use custom logic

        return {
            "question": request.question,
            "answer": response["answer"],
            "sources": response["sources"]   # ✅ include sources
        }

    except Exception as e:
        return {"error": str(e)}


import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)    