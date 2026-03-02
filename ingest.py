print("📥 Starting ingestion...")
import os
import re
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import chromadb
from pypdf import PdfReader
import docx

# config files
DATA_DIR = "data"
UPLOAD_DIR = "uploads"
VECTOR_DB_DIR = "vectordb"
TRACK_FILE = "ingested_files.txt"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx"]

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# file loader
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

#tracking
def load_ingested_files():
    if not os.path.exists(TRACK_FILE):
        print("No tracking file found, creating new one...")
        open(TRACK_FILE, "w").close()  # create empty file
        return set()

    with open(TRACK_FILE, "r") as f:
        files = set(line.strip() for line in f.readlines())
        print(f"Already ingested files: {files}")
        return files


def save_ingested_file(filename):
    with open(TRACK_FILE, "a") as f:
        f.write(filename + "\n")

def load_documents():
    documents = []

    print("Loading NEW files only...")

    ingested_files = load_ingested_files()
    new_files_count = 0

    all_dirs = [DATA_DIR, UPLOAD_DIR]

    for directory in all_dirs:
        if not os.path.exists(directory):
            print(f"Folder not found: {directory}")
            continue

        files = os.listdir(directory)

        for filename in tqdm(files, desc=f"Processing {directory}"):
            filepath = os.path.join(directory, filename)

            print(f"\nChecking file: {filename}")

            try:
                # Skip unsupported
                if not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    print(f"⏭ Skipped (unsupported): {filename}")
                    continue

                # Skipping already ingested
                if filename in ingested_files:
                    print(f"⏭ Already ingested: {filename}")
                    continue

                print(f"Processing NEW file: {filename}")

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

                    save_ingested_file(filename)
                    new_files_count += 1
                    print(f"Saved to tracking: {filename}")

                else:
                    print(f"Empty content: {filename}")

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print(f"\n New files added: {new_files_count}")
    print(f"Documents to process: {len(documents)}")

    return documents

# ingest
def ingest():
    documents = load_documents()

    if not documents:
        print("No new documents to ingest!")
        return

    # Chunking
    parser = SimpleNodeParser.from_defaults(
        chunk_size=500,
        chunk_overlap=50
    )

    print("\n Chunking documents...")
    nodes = parser.get_nodes_from_documents(documents)

    print(f"Total chunks: {len(nodes)}")

    # Embedding
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Persistent DB
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    collection = chroma_client.get_or_create_collection(
        name="rag_collection"
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("\n Adding new data to vector DB...")

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("\n Ingestion completed (no duplicates)!")


#run
if __name__ == "__main__":
    ingest()