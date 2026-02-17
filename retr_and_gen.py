import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI

import chromadb

load_dotenv()

# config files
VECTOR_DB_DIR = "vectordb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-flash-latest"

# Embedding model
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL
)

# persistent Chroma DB
chroma_client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR
)

 # collection folder's content
collection = chroma_client.get_collection(
    name="rag_collection"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# loading index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

llm = GoogleGenAI(
    model=LLM_MODEL,
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# Query engine
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3 
)

# Query
query = "Summarize the key ideas from both PDFs"
response = query_engine.query(query)

print("\n🧠 Answer:")
print(response)
