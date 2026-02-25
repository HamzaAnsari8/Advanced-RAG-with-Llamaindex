import os
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Document

import chromadb

load_dotenv()

# config file
DATA_DIR = "data"            #folder with files
VECTOR_DB_DIR = "vectordb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# loading pdf
documents = SimpleDirectoryReader(
    DATA_DIR,
    recursive=True,
    filename_as_id=True
).load_data()


cleaned_documents = []

for doc in documents:
    clean_text = doc.text.encode("utf-8", errors="ignore").decode()
    metadata = doc.metadata or {}
    metadata["file_name"]=metadata.get("file_name","").lower()
    cleaned_documents.append(Document(text=clean_text,metadata=metadata))

documents = cleaned_documents

print(f" Loaded {len(documents)} document chunks")

# calling embeding model
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL
)

# creating chroma DB visibily
chroma_client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR
)

collection = chroma_client.get_or_create_collection(
    name="rag_collection"
)

#naming a folder "collection" in vectordb 
vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# indexing-
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    storage_context=storage_context
)

print("PDF ingestion completed")
print(f"Vector DB saved at: {VECTOR_DB_DIR}")
 