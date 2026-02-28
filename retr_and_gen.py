print("retr_and_gen.py loaded")
import os
import re
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.google_genai import GoogleGenAI

# Load env
load_dotenv()

# Check API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY missing")

# Config
VECTOR_DB_DIR = "vectordb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-flash-latest"

# Globals
index = None
llm = None
chat_history = []


#Lazy loader 
def load_rag():
    global index, llm

    try:
        print("Loading RAG...")

        # Embedding
        embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL
        )

        # Chroma DB
        chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_DIR
        )

        # SAFE collection load
        try:
            collection = chroma_client.get_collection(name="rag_collection")
        except:
            print("Collection not found, creating new one")
            collection = chroma_client.get_or_create_collection(name="rag_collection")

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # Index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        # LLM
        llm = GoogleGenAI(
            model=LLM_MODEL,
            api_key=api_key,
            temperature=0.2,
            max_tokens=256
        )

        print("RAG Loaded")

        return index, llm

    except Exception as e:
        print("ERROR IN load_rag:", e)
        raise e

#Main function
def ask_question(query: str):
    print("ask question called")
    print("starting processing")

    global chat_history

    try:
        # Load RAG only when needed
        index, llm = load_rag()

        # Build context
        full_query = ""
        for q, a in chat_history[-2:]:
            full_query += f"User: {q}\nAssistant: {a}\n"

        full_query += f"User: {query}\nAssistant:"

        filters = None

        # Detect filename
        match = re.search(r"\b\w+\.(txt|pdf|docx)\b", query.lower())
        if match:
            file_name = match.group().lower()

            filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(
                        key="file_name",
                        value=file_name
                    )
                ]
            )

        # Query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,
            response_mode="compact",
            filters=filters
        )

        response = query_engine.query(full_query)

        # Extract answer
        answer_text = response.response

        # Extract sources
        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                file_name = node.metadata.get("file_name", "Unknown")
                if file_name not in sources:
                    sources.append(file_name)

        # Save history
        chat_history.append((query, answer_text))
        chat_history = chat_history[-3:]

        return {
            "answer": answer_text,
            "sources": sources
        }

    except Exception as e:
        print(" ERROR IN ask_question:", e)
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }