# LangGraph Chatbot

A Streamlit-based chatbot that combines LangGraph agent orchestration, Gemini LLMs, PDF-grounded retrieval, web search, and calculator tools. Each user has isolated chat threads, and conversations can be resumed from PostgreSQL-backed checkpoints.

## Overview

This project turns a single Streamlit app into a multi-tool AI assistant:

- Uses LangGraph to orchestrate a chat workflow and optional tool-calling loop
- Uses Gemini models for conversational reasoning and embeddings
- Adds a RAG layer for uploaded PDF documents using FAISS and LangChain document splitting
- Supports web search via DuckDuckGo and simple arithmetic via a calculator tool
- Stores chat history and thread ownership in PostgreSQL
- Keeps user-specific chats isolated by requiring authenticated Streamlit users

The app is split into two main files:

- `backend.py` — orchestration, tool definitions, memory management, and persistence
- `frontend.py` — authentication, thread management, PDF upload UI, and chat interface

## Key Features

- PDF Q&A over uploaded documents
- One indexed PDF per conversation thread
- Persistent chat history across sessions
- Thread-specific state and memory summaries
- Google login / authenticated user flow in Streamlit
- Web search tool for open-domain lookup
- Calculator tool for basic arithmetic
- PostgreSQL-based checkpointing for resumeable conversations

## Architecture

### Workflow

The LangGraph application is built as a graph with these nodes:

- `chat_node` — sends the latest user message and system instructions to Gemini
- `tools` — executes tool calls such as web search, calculator, or PDF retrieval
- `summarize` — condenses long conversations into a concise memory summary

The routing logic does the following:

- If the LLM requests a tool call, the graph routes to `tools`
- If the conversation grows beyond a threshold, it triggers summarization
- Otherwise the chat ends cleanly

### PDF Retrieval

When a PDF is uploaded:

1. The file is written to a temporary location
2. `PyPDFLoader` extracts the pages
3. `RecursiveCharacterTextSplitter` splits pages into chunks
4. A FAISS vector store is created with Gemini embeddings
5. A retriever is stored per thread in memory for later retrieval

Then, when the LLM asks a question related to the uploaded PDF, the `rag_tool` searches the indexed chunks and returns relevant context for grounding the answer.

### Persistence

The app uses:

- `PostgresSaver` from `langgraph.checkpoint.postgres`
- PostgreSQL connection details from the `DB_URI` environment variable
- A table named `app_user_threads` to map `user_id` to `thread_id`

This means chat states, thread history, and resumable conversations are stored outside the browser session.

## Repository Layout

```text
.
├── .gitignore
├── README.md
├── backend.py
├── frontend.py
└── .env.example (if added in your local setup)
```

## Tech Stack

- Python
- Streamlit
- LangGraph
- LangChain
- Google Gemini (`ChatGoogleGenerativeAI`, embeddings API)
- FAISS
- PyPDF
- PostgreSQL
- DuckDuckGo Search

## Prerequisites

Before running the app, make sure you have:

- Python 3.10+
- A PostgreSQL instance available
- A Google Gemini API key
- A Streamlit environment that supports authenticated users (the app checks `st.user.is_logged_in` and uses `st.login()`/`st.logout()`)

## Environment Variables

Create a `.env` file in the project root with the following values:

```bash
GOOGLE_API_KEY=your_google_gemini_api_key
DB_URI=postgresql://username:password@host:5432/database_name
```

Notes:

- `GOOGLE_API_KEY` is required because the app initializes Gemini models in `backend.py`.
- `DB_URI` is required for LangGraph Postgres checkpointing and thread ownership storage.
- If you deploy on Streamlit Cloud or another hosted environment, ensure your auth and environment configuration match the app's expectations.

## Dependencies

Install the required Python packages:

```bash
pip install streamlit langgraph langchain-core langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf python-dotenv psycopg psycopg-pool duckduckgo-search
```

Depending on your environment, you may also need a PostgreSQL client library or additional system packages for your operating system.

## Running the App

1. Clone the repository
2. Create and activate a virtual environment
3. Install the dependencies
4. Add your `.env` file
5. Start PostgreSQL and ensure the `DB_URI` target database exists
6. Launch the app:

```bash
streamlit run frontend.py
```

Then open the provided local URL in your browser.

## Usage

### 1. Sign in

The app requires authentication before you can chat. If a user is not logged in, it shows a login screen and calls `st.login()`.

### 2. Create or select a chat thread

Each authenticated user can create new chat sessions and switch between previous conversations. The app stores thread IDs in PostgreSQL and enforces ownership checks before access.

### 3. Upload a PDF

Use the sidebar uploader to add a PDF. Once processed, the app indexes the file for the active thread. The current implementation supports one indexed document per conversation thread.

### 4. Ask questions

You can ask:

- Questions directly about the uploaded PDF
- General knowledge questions using the LLM
- Arithmetic calculations using the built-in calculator
- Web-based lookup via the DuckDuckGo search tool

### 5. Resume conversations

Because checkpoints are stored in PostgreSQL, users can return to prior chats and continue the history without losing context.

## Current Limitations

- Only one PDF is indexed per chat thread
- Retrieval is in-memory per thread, which is fine for local use but not a distributed long-term storage design
- Web search is implemented with DuckDuckGo and may vary by query and region
- The app is tightly coupled to Streamlit authentication and Google-based identity in its current form

## Security and Data Notes

- User-to-thread access is enforced with `user_owns_thread()` checks before reading or writing a thread
- PDF documents are stored in a retriever cache keyed by `thread_id`
- The app relies on environment variables for sensitive credentials and database connections
- Conversation memory summaries are generated automatically and stored as part of the graph state

## Example Workflow

```text
User logs in
↓
Creates a chat thread
↓
Uploads a PDF
↓
LLM answers grounded questions using the PDF retriever
↓
User asks a follow-up question
↓
Thread state is restored from PostgreSQL checkpoints
```

## Development Notes

This repo is intentionally compact and showcases a practical pattern for building a multi-tool AI assistant using LangGraph and Streamlit. It is an excellent starting point for adding:

- multi-document PDF ingestion
- document deletion or replacement
- richer source citations
- user profile management
- custom auth flows
- more advanced tool orchestration
- deployment to Docker or cloud infrastructure

## License

This project does not currently declare a license in the repository. If you plan to distribute or reuse it publicly, add an explicit license file such as MIT or Apache-2.0.

## Summary

`langgraph-chatbot` is a full-stack demo application that demonstrates how to combine a Streamlit frontend, LangGraph orchestration, Gemini AI, document retrieval, and persistent chat history into a useful assistant experience.

If you are evaluating the repo for learning or customization, the most important implementation files are:

- `backend.py` for reasoning and tools
- `frontend.py` for UI and thread logic
- `README.md` for setup and onboarding

This README was updated to reflect the actual repository implementation and setup requirements.
