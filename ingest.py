print("📥 Starting ingestion...")
import os
import re
import chromadb
from tqdm import tqdm
from pypdf import PdfReader
import docx
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

# CONFIG
DATA_DIR = "data"
VECTOR_DB_DIR = "vectordb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]

# CLEAN TEXT
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# FILE LOADERS
def load_txt(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_docx(filepath):
    document = docx.Document(filepath)
    return "\n".join([para.text for para in document.paragraphs])

# LOAD DOCUMENTS
def load_documents():
    documents = []
    if not os.path.exists(DATA_DIR):
        print("data folder not found")
        return documents
    files = os.listdir(DATA_DIR)
    for filename in tqdm(files, desc="Processing documents"):
        filepath = os.path.join(DATA_DIR, filename)

        if not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            continue

        try:
            text = ""
            if filename.lower().endswith(".txt"):
                text = load_txt(filepath)

            elif filename.lower().endswith(".pdf"):
                text = load_pdf(filepath)

            elif filename.lower().endswith(".docx"):
                text = load_docx(filepath)

            text = clean_text(text)
            if text:
                documents.append(
                    Document(
                        text=text,
                        metadata={"file": filename}
                    )
                )
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    print(f"📄 Documents loaded: {len(documents)}")
    return documents

# INGEST
def ingest():
    documents = load_documents()
    if not documents:
        print("⚠️ No documents found")
        return

    parser = SimpleNodeParser.from_defaults(
        chunk_size=500,
        chunk_overlap=50
    )
    nodes = parser.get_nodes_from_documents(documents)
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL
    )
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR
    )
    collection = chroma_client.get_or_create_collection(
        name="rag_collection"
    )
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex(
        [],
        storage_context=storage_context,
        embed_model=embed_model
    )
    index.insert_nodes(nodes)
    print("Ingestion completed!")

# RUN
if __name__ == "__main__":
    ingest()