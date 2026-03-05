Advanced RAG with LlamaIndex

An Advanced Retrieval-Augmented Generation (RAG) system built with LlamaIndex, ChromaDB, HuggingFace embeddings, Gemini LLM, and Streamlit.
The system allows users to upload documents and ask questions using natural language. It retrieves relevant document content and generates accurate answers using a large language model.
This implementation includes Hybrid Retrieval, Cross-Encoder Reranking, and Smart Filename Detection for more accurate document search.

Project Overview
Traditional LLMs cannot access private documents.
RAG solves this problem by combining:
Vector Search
Keyword Search (BM25)
LLM reasoning

This project implements an advanced hybrid retrieval pipeline that improves accuracy and document understanding.

Features

Document Question Answering

Ask questions about uploaded documents such as:

PDFs

TXT files

DOCX files

Example queries:

Summarize AI FreeBook

What is written in Hamza Ansari resume?

Explain Deep Learning from the document

Hybrid Search (Vector + BM25)
The system combines two retrieval techniques:
Vector Search

Semantic similarity using embeddings
BM25 Search

Keyword-based retrieval
This improves recall and ensures relevant content is retrieved.

Cross-Encoder Reranking
Retrieved documents are reranked using a cross-encoder model

cross-encoder/ms-marco-MiniLM-L-6-v2 
This step improves the quality of the final context sent to the LLM.

Smart Filename Query Detection

Users can search documents using:

Exact filename:
summarize ai_FreeBook.pdf 

Filename without extension:
summarize ai freebook 

Partial name:
summarize resume

The system automatically detects the relevant document.
Conversational Memory

The system keeps a short chat history context, allowing follow-up questions.
Example:
User: summarize the AI book User: explain deep learning from it 

Source Attribution:
The system displays which document the answer came from.
Example:
Sources: ai_FreeBook.pdf 

System Architecture:

User Query │ 
   ▼
Streamlit Interface 
   ▼
Query Processing
   ▼
Hybrid Retrieval (Vector Search + BM25)
   ▼
Cross Encoder Reranking 
   ▼ 
Top Relevant Context
   ▼ 
Gemini LLM 
   ▼ 
Final Answer + Sources 

Project Structure:
Advanced-RAG-with-LlamaIndex │ ├── app.py ├── ingest.py ├── retr_and_gen.py ├── vectordb/ ├── uploads/ ├── requirements.txt ├── .env └── README.md 

app.py
Streamlit user interface for document upload and chat.

ingest.py
Processes uploaded documents and stores embeddings in ChromaDB.

retr_and_gen.py

Core RAG pipeline containing:

Hybrid retrieval
Reranking

LLM generation
Filename matching

vectordb/
Stores the Chroma vector database.

uploads/
Directory for uploaded documents.

Technologies Used:

LlamaIndex:
Framework for building RAG pipelines.

ChromaDB:
Vector database used for storing document embeddings.

HuggingFace Embeddings:
sentence-transformers/all-MiniLM-L6-v2 

Gemini LLM:
Google Generative AI used for answer generation.
Cross Encoder Reranker
cross-encoder/ms-marco-MiniLM-L-6-v2 

Streamlit:
Interactive web interface.

Installation

Clone the repository
git clone https://github.com/yourusername/Advanced-RAG-with-LlamaIndex.git cd Advanced-RAG-with-LlamaIndex 

Create virtual environment
python -m venv venv 
Activate environment
Windows
venv\Scripts\activate 
Linux / Mac
source venv/bin/activate 

Install dependencies
pip install -r requirements.txt 

Environment Setup

Create a .env file:
GOOGLE_API_KEY=your_google_api_key 
You can generate the API key from:
https://aistudio.google.com/app/apikey 

Running the Application

Start the Streamlit server:
streamlit run app.py 
Open in browser:
http://localhost:8501 

How It Works

Step 1 — Upload Documents
Supported formats:
PDF
TXT
DOCX
Documents are processed and stored in the vector database.

Step 2 — Document Chunking
Documents are split into smaller chunks for better retrieval.

Step 3 — Embedding Generation
Each chunk is converted into vector embeddings.

Step 4 — Hybrid Retrieval
The system retrieves relevant chunks using:
Vector similarity
BM25 keyword search

Step 5 — Reranking
A cross-encoder reranks retrieved documents to select the most relevant context.

Step 6 — LLM Answer Generation
The Gemini model generates answers based only on the retrieved context.

Example Queries:

Context queries:
What is deep learning? Explain hidden markov models 

Filename queries:
summarize ai_FreeBook.pdf summarize resume what is written in hamza ansari resume 
Follow-up queries
Explain reinforcement learning from that book 

Limitations:
Free Gemini API has rate limits and quotas
Large documents may require chunk size optimization
Context window is limited by model token limits
Future Improvements

Possible enhancements:
Multi-document reasoning
Metadata filtering
Query rewriting
Streaming responses
Document highlighting
Vector database scaling

Author
Hamza Ansari
BE Computer Science Engineering
Specialization: Artificial Intelligence & Machine Learning

License

This project is for educational and research purposes.