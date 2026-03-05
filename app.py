import streamlit as st
import os
import time
from retr_and_gen import ask_question
import ingest

# ---------------- AUTO INGEST FOR DEPLOYMENT ----------------
VECTOR_DB_DIR = "vectordb"

if not os.path.exists(VECTOR_DB_DIR):
    print("Vectordb not found. Running ingestion...")
    ingest.ingest()
# ------------------------------------------------------------

# PAGE
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.markdown("Ask questions from your documents")

st.sidebar.markdown("""
### 📚 Available Documents

- AI Freebook
- Web Development Ebook
- Ottoman Empire History
- Research Proposal
- Sample Text
""")

# SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# USER INPUT
query = st.chat_input("Ask something...")

if query:
    st.session_state.chat_history.append(("user", query))

    with st.spinner("Thinking..."):
        time.sleep(2)  # prevents Gemini rate limit
        response = ask_question(query)

    answer = response["answer"]
    sources = response["sources"]

    st.session_state.chat_history.append(("bot", answer, sources))

# DISPLAY CHAT
for item in st.session_state.chat_history:

    if item[0] == "user":
        with st.chat_message("user"):
            st.write(item[1])

    else:
        answer = item[1]
        sources = item[2]

        with st.chat_message("assistant"):

            # STREAMING ANSWER
            placeholder = st.empty()
            streamed_text = ""

            for word in answer.split():
                streamed_text += word + " "
                placeholder.markdown(streamed_text)
                time.sleep(0.02)

            # SOURCES
            if sources:
                st.markdown("### 📚 Sources")

                src_placeholder = st.empty()
                src_text = ""

                for src in sources:
                    src_text += f"- {src}\n"
                    src_placeholder.markdown(src_text)
                    time.sleep(0.05)