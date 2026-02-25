import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters,ExactMatchFilter
from llama_index.llms.google_genai import GoogleGenAI
import re
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

# vector database Chroma DB
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
    temperature=0.2,
    max_token=256
)

chat_history = []
# Query
while True:
    query = input("\nAsk something (or type 'exit'): ")

    if query.lower() == "exit":
        break

    try:
       # Build context using history
       full_query = ""
       for q, a in chat_history[-2:]:
           full_query += f"User: {q}\nAssistant: {a}\n"

       full_query += f"User: {query}\nAssistant:"

       filters = None

       # detect file name dynamically
       match = re.search(r"\b\w+\.(txt|pdf|docx)\b", query.lower())

       if match:
            file_name = match.group().lower()

            filters = MetadataFilters(
                  filters=[
            ExactMatchFilter(key="file_name", value=file_name)
            ]
        )

       # Query engine
       query_engine = index.as_query_engine(
           llm=llm,
           similarity_top_k=3,
           streaming=True,
           response_mode="compact",
           filters=filters)

       response = query_engine.query(query)

       print("\nAnswer:\n")

       answer_text = ""
       for token in response.response_gen:
           print(token, end="", flush=True)
           answer_text += token

       print("\n")

       # Save history (last 3 only)
       chat_history.append((query, answer_text))
       chat_history = chat_history[-3:]

       # Show souces of documents
       print("\nSources:\n")

       if hasattr(response,"source_nodes") and response.source_nodes:
          for i, node in enumerate(response.source_nodes):
             file_name = node.metadata.get("file_name", "Unknown")
             print(f"{i+1}. {file_name}")

             # optional preview
             print("Preview:", node.text[:120])
             print("-" * 50)

       else: 
         print("No source found")    

    except Exception as e:
       print("Error:", e)