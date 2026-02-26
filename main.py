from fastapi import FastAPI
from retr_and_gen import ask_question

app=FastAPI()

@app.get("/")

def home():
    return{"messages":"working"}