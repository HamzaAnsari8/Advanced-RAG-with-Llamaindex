print("🚀 Advanced retr_and_gen.py loaded")
import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()

# CONFIG FILES
VECTOR_DB_DIR = "vectordb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-flash-latest"

api_key = os.getenv("GOOGLE_API_KEY")

index = None
llm = None
bm25 = None
documents_store = []
chat_history = []

# LOAD RAG
def load_rag():
    global index, llm, bm25, documents_store

    try:
        print("Loading Advanced RAG...")

        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

        # Chroma DB
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        collection = chroma_client.get_or_create_collection("rag_collection")

        vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        # BM25
        print("📚 Preparing BM25...")
        all_docs = collection.get()["documents"]

        documents_store = all_docs
        tokenized_docs = [doc.split() for doc in all_docs]

        bm25 = BM25Okapi(tokenized_docs)

        # LLM
        llm = GoogleGenAI(
            model=LLM_MODEL,
            api_key=api_key,
            temperature=0.3,
            max_tokens=1024
        )

        print("RAG Ready")

        return index, llm, bm25

    except Exception as e:
        print("ERROR in load_rag:", e)
        raise e

#RERANK
print("Loading reranker...")

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank(query, docs):
    pairs = [[query, doc] for doc in docs]

    inputs = tokenizer(
        pairs, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze()

    scored_docs = list(zip(docs, scores.tolist()))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:3]]

#MAIN FUNCTION
def ask_question(query: str):
    global chat_history

    try:
        print("\n🔍 Query:", query)

        index, llm, bm25 = load_rag()

        # CHAT HISTORY
        context_query = query
        if len(chat_history) > 0:
            last_q, last_a = chat_history[-1]
            context_query = last_q + " " + query

        # VECTOR SEARCH
        vector_results = index.as_retriever(
            similarity_top_k=3
        ).retrieve(context_query)

        vector_docs = [node.text for node in vector_results]

        # BM25 SEARCH
        tokenized_query = context_query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        top_bm25_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:3]

        bm25_docs = [documents_store[i] for i in top_bm25_idx]

        # HYBRID
        combined_docs = list(set(vector_docs + bm25_docs))
        print("📊 Hybrid docs:", len(combined_docs))

        # RERANK
        top_docs = rerank(context_query, combined_docs)

        context = "\n\n".join(top_docs)

        #BETTER PROMPT
        prompt = f"""
You are an intelligent AI assistant.

Use ONLY the provided context to answer.
If answer is not found, say "Not enough information".

Context:
{context}

Question:
{query}

Answer clearly:
"""

        print("Generating...")

        response = llm.complete(prompt)
        answer_text = str(response)

        # SOURCES
        sources = list(set([doc[:150] for doc in top_docs]))

        # SAVE HISTORY
        chat_history.append((query, answer_text))
        chat_history = chat_history[-3:]

        return {
            "answer": answer_text,
            "sources": sources
        }

    except Exception as e:
        print("ERROR:", e)
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }