# LangGraph Chatbot

This project is a full-featured Streamlit chatbot built with LangGraph and Google Gemini that helps users interact with documents, the web, and built-in tools in one place. It supports user authentication, persistent chat threads, and separate conversation histories so each user can maintain their own context. Users can upload PDF documents, ask questions about their contents, and receive answers using a retrieval-augmented workflow that splits documents into chunks, embeds them with Gemini, and stores them in a FAISS vector index for semantic search. The assistant can also answer questions using web search through DuckDuckGo and perform simple arithmetic calculations when needed. It includes short-term conversation memory: as a chat grows, the workflow summarizes earlier messages into a concise memory and removes older messages from the active history to help keep context and token use smaller. PostgreSQL-backed checkpoints persist chat state across sessions.

## Features

- Upload and ask questions about PDFs
- Use Gemini-powered chat with tool calling
- Summarize earlier conversation messages into compact short-term memory as the chat grows
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

- `backend.py` contains the LangGraph workflow, tool definitions, PDF ingestion, conversation summarization, and Postgres checkpointing.
- `frontend.py` handles authentication, chat UI, PDF upload, and thread management.
- Uploaded PDFs are split into chunks, embedded with Gemini, and stored in a FAISS vector index per thread.
- When the conversation exceeds the message threshold, the workflow summarizes earlier messages and removes them from the active message history to limit its size.
- The assistant can answer questions from the document, search the web, or do arithmetic when needed.

## Notes

- The app requires a logged-in user session.
- Only one PDF is indexed per chat thread in the current implementation.
- The database must be available before launching the app.
