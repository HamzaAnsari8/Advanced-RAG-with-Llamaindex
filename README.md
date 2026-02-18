# 📚 RAG with LlamaIndex (Gemini API)

This project implements a **Retrieval-Augmented Generation (RAG)** system using **LlamaIndex**, **Google Gemini API**, and **ChromaDB**. It allows users to query their own documents by retrieving relevant context from a vector database and generating accurate, context-aware answers using Gemini.

The project is designed to be simple, clean, and easy to understand, making it suitable for learning, academic projects, and real-world applications.


# 🚀 Features
- Document ingestion and preprocessing  
- Text chunking using LlamaIndex  
- Embedding generation using Gemini API  
- Vector storage and retrieval with ChromaDB  
- Semantic search and context-aware answer generation  
- Modular and extensible design  


# 🧠 Project Structure
RAG-with-LlamaIndex-Gemini/  
├── data/                  # Input documents (PDF / TXT / DOCX)  
├── chroma_db/             # ChromaDB vector storage  
├── ingest.py              # Document ingestion & indexing  
├── retr_and_gen.py        # Retrieval and response generation  
├── requirements.txt       # Python dependencies  
├── .env                   # Environment variables  
└── README.md              # Project documentation  


# 🛠️ Tech Stack
- Python 3.9+  
- LlamaIndex  
- Google Gemini API  
- ChromaDB  
- python-dotenv  


# 🔐 Environment Setup
Create a `.env` file in the project root directory and add your Gemini API key:

GOOGLE_API_KEY="your_gemini_api_key_here"  


# 📦 Installation
1. Clone the repository  
git clone https://github.com/HamzaAnsari8/rag-llamaindex-gemini.git  
cd rag-llamaindex-gemini  

2. Create and activate a virtual environment  
python -m venv venv  
source venv/bin/activate      # Linux / macOS  
venv\Scripts\activate         # Windows  

3. Install dependencies  
pip install -r requirements.txt  



# 📥 Document Ingestion (ingest.py)
This script:
- Loads documents from the `data` directory  
- Splits documents into chunks using LlamaIndex  
- Generates embeddings using Gemini  
- Stores embeddings in ChromaDB  

Run ingestion whenever documents are added or updated:

python ingest.py  


# 🔍 Retrieval and Generation (retr_and_gen.py)
This script:
- Accepts a user query  
- Retrieves relevant document chunks from ChromaDB  
- Sends the retrieved context along with the query to Gemini  
- Generates a final, context-aware answer  

Run the RAG pipeline:

python retr_and_gen.py  

Example:  
Enter your query: What is Retrieval Augmented Generation?  
Answer: Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with large language models to produce accurate and grounded responses.


# 🔁 RAG Workflow
1. Documents are ingested and indexed  
2. User submits a query  
3. Relevant document chunks are retrieved from ChromaDB  
4. Gemini generates an answer using the retrieved context  


# 📌 Use Cases
- Chat with PDFs and text documents  
- Internal knowledge base assistant  
- Research and academic support  
- Enterprise document search  
- AI-powered FAQ systems  


# ⚠️ Notes
- Place documents inside the `data` directory before ingestion  
- Re-run `ingest.py` when documents change  
- Gemini API usage may incur costs depending on usage  



# 🔮 Future Enhancements
- Web UI using Streamlit or FastAPI  
- Conversational memory  
- Streaming responses  
- Metadata-based filtering  
- Cloud-hosted vector databases  


# 👨‍💻 Author
Hamza Ansari
GitHub: https://github.com/HamzaAnsari8 
