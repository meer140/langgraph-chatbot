# LangGraph Chatbot

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C1C1C?style=for-the-badge&logo=react&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1F8FFF?style=for-the-badge&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8A2BE2?style=for-the-badge&logo=google&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-00C7B7?style=for-the-badge&logo=postgresql&logoColor=white)
![PyPDF](https://img.shields.io/badge/PyPDF-4B8BBE?style=for-the-badge&logo=python&logoColor=white)

A Streamlit chatbot built with LangGraph and Google Gemini. Ask questions about uploaded PDFs, search the web, and use a calculator—all in one chat app. Conversation state and PDF embeddings are persisted in PostgreSQL.

## Features

- Upload PDFs and ask questions grounded in their contents
- Store and search document embeddings in PostgreSQL with **pgvector** (replacing FAISS)
- Keep PDF retrieval scoped to the current conversation
- Use Gemini-powered chat with tool calling
- Summarize older messages into short-term conversation memory
- Retrieve long-term memories across conversations
- Search the web with DuckDuckGo
- Perform basic arithmetic with a calculator tool
- Keep separate chat histories per user with PostgreSQL checkpoints

## Tech stack

- Python and Streamlit
- LangGraph and LangChain
- Google Gemini for chat and embeddings
- PostgreSQL with the pgvector extension, using LangChain's `PGVector` integration
- PyPDF
- DuckDuckGo Search

## Setup

1. Create and activate a Python virtual environment, then install the dependencies:

   ```bash
   pip install streamlit langgraph langchain-core langchain-community langchain-google-genai langchain-text-splitters langchain-postgres pypdf python-dotenv 'psycopg[binary]' psycopg-pool duckduckgo-search
   ```

2. Set up a PostgreSQL database with the **pgvector extension** installed and enabled. The application does not create the extension automatically. For example, with a database role that has permission to create extensions:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. Create a `.env` file in the project directory:

   ```dotenv
   GOOGLE_API_KEY=your_google_gemini_api_key
   DB_URI=postgresql://username:password@host:5432/database_name
   ```

   `DB_URI` is used for both LangGraph checkpointing and the PostgreSQL vector store.

4. Start the app:

   ```bash
   streamlit run frontend.py
   ```

## Project structure

```text
.
├── backend.py
├── frontend.py
├── README.md
└── .gitignore
```

## How it works

- `backend.py` defines the LangGraph workflow, tools, PDF ingestion and retrieval, conversation summarization, and PostgreSQL checkpointing.
- `frontend.py` provides the authentication, chat interface, PDF upload, and thread management.
- Uploaded PDFs are split into chunks, embedded with Gemini, and stored in PostgreSQL through `PGVector`. Retrieval filters by conversation thread so results come from the current chat's documents.
- As a conversation grows, earlier messages are summarized and removed from the active message history.
- The assistant can retrieve information from uploaded documents, search the web, or perform arithmetic when appropriate.

## Pipeline

![LangGraph workflow pipeline](pipeline.svg)

## Notes

- A logged-in user session is required.
- Multiple PDFs can be uploaded to a conversation.
- PostgreSQL, with pgvector enabled, must be available before launching the app.
