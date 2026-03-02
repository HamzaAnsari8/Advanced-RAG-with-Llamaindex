🤖 Advanced RAG Chatbot (LlamaIndex + Gemini + Streamlit)

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask questions based on their content.
Built using:

LlamaIndex

Google Gemini (LLM)

ChromaDB (Vector Database)

Streamlit (UI)

HuggingFace Embeddings

🚀 Features

💬 Chatbot Capabilities

Ask questions from your own documents

Context-aware responses using RAG pipeline

Streaming responses (word-by-word like ChatGPT)

Source citation support

📂 Document Support

PDF (.pdf)

Text files (.txt)

Word documents (.docx)

⚡ Advanced Features

Persistent vector database (ChromaDB)

No duplicate embeddings (tracked ingestion)

Upload documents from UI

Supports multiple documents

Efficient chunking + embedding

🧠 Architecture Overview

User Query → Retriever → Relevant Chunks → LLM (Gemini) → Response + Sources

Pipeline:

Documents are loaded

Cleaned & preprocessed

Chunked into smaller pieces

Converted into embeddings

Stored in ChromaDB

Query retrieves relevant chunks

Gemini generates final answer

📁 Project Structure

RAG-with-LlamaIndex/ │ ├── app.py # Streamlit UI (chat + upload + streaming) ├── ingest.py # Data ingestion & embedding pipeline ├── retr_and_gen.py # Retrieval + LLM response logic ├── requirements.txt # Dependencies │ ├── data/ # Static documents ├── uploads/ # User uploaded files ├── vectordb/ # Persistent vector database ├── ingested_files.txt # Tracks processed files │ └── README.md

⚙️ Installation

1️⃣ Clone Repository

git clone 
cd RAG-with-LlamaIndex

2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate (Windows)

3️⃣ Install Dependencies

pip install -r requirements.txt

🔑 Environment Variables

Create .env file:
GOOGLE_API_KEY=your_gemini_api_key

📥 Data Ingestion

Step 1: Add files

Put files inside: 

data/ (manual)

OR upload via UI → uploads/

Step 2: Run ingestion

python ingest.py
✅ This will:

Convert documents → embeddings

Store in vector DB

Avoid duplicates automatically

💻 Run Application

streamlit run app.py
Open in browser:
http://localhost:8501

📤 Upload via UI

Upload PDF / TXT / DOCX from sidebar

Files saved in uploads/

Then run:

python ingest.py

🔄 Streaming Output

Answers appear word-by-word

Sources appear line-by-line

Improves user experience (real-time feel)

🧩 Key Components

🔹 LlamaIndex

Handles:

Document parsing

Chunking

Retrieval pipeline

🔹 ChromaDB

Stores embeddings persistently

Fast similarity search

🔹 HuggingFace Embeddings

Model used: sentence-transformers/all-MiniLM-L6-v2

🔹 Gemini (LLM)

Generates final answers

Uses retrieved context

🛡️ Duplicate Handling

✔ No duplicate embeddings
How it works:

ingested_files.txt tracks processed files

Already ingested files are skipped

⚠️ Notes

If you delete vectordb/ → all embeddings lost

If you delete ingested_files.txt → system re-ingests all files

Always run ingest.py after uploading new documents

🔥 Future Improvements

Auto-ingestion on upload (no manual step)

PDF preview in UI

Chat history persistence

Multi-user support

API deployment (FastAPI)

Docker support

Cloud deployment (AWS/GCP)

🧪 Example Queries

What is AI?
Summarize this document
Explain key concepts from research paper

👨‍💻 Author

Hamza Ansari

⭐ Support

If you like this project:
⭐ Star the repo
📢 Share it