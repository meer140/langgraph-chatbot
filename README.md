# LangGraph Chatbot

A Streamlit app powered by LangGraph and Gemini for PDF Q&A, web search, calculator tools, and persistent chat threads.

## Features

- Upload and ask questions about PDFs
- Use Gemini-powered chat with tool calling
- Search the web with DuckDuckGo
- Run basic arithmetic through a calculator tool
- Keep separate chat histories per user with PostgreSQL checkpoints

## Tech Stack

- Python
- Streamlit
- LangGraph
- LangChain
- Google Gemini
- FAISS
- PyPDF
- PostgreSQL

## Setup

1. Create a virtual environment and install dependencies:

```bash
pip install streamlit langgraph langchain-core langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf python-dotenv psycopg psycopg-pool duckduckgo-search
```

2. Add a `.env` file:

```bash
GOOGLE_API_KEY=your_google_gemini_api_key
DB_URI=postgresql://username:password@host:5432/database_name
```

3. Start the app:

```bash
streamlit run frontend.py
```

## Project Structure

```text
.
├── backend.py
├── frontend.py
├── README.md
├── .gitignore
```

## How it works

- `backend.py` contains the LangGraph workflow, tool definitions, PDF ingestion, and Postgres checkpointing.
- `frontend.py` handles authentication, chat UI, PDF upload, and thread management.
- Uploaded PDFs are split into chunks, embedded with Gemini, and stored in a FAISS vector index per thread.
- The assistant can answer questions from the document, search the web, or do arithmetic when needed.

## Notes

- The app requires a logged-in user session.
- Only one PDF is indexed per chat thread in the current implementation.
- The database must be available before launching the app.
