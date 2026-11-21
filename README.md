# LangGraph Chatbot

[![Live demo](https://img.shields.io/badge/Live-Demo-FF4B4B?style=for-the-badge)](https://langgraph-chatbot-u5li.onrender.com/)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C1C1C?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-1F8FFF?style=for-the-badge&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)

A full-stack AI chatbot built with **Streamlit, LangGraph, LangChain, and Google Gemini**. It supports document-grounded chat, web search, arithmetic, persistent conversations, short-term summaries, and long-term user memories.

## Features

- **PDF question answering** — upload one or more PDFs and ask questions grounded in their contents
- **Thread-scoped RAG** — Gemini embeddings and pgvector retrieve chunks only from the active conversation
- **Persistent chat history** — LangGraph checkpoints and conversation ownership are stored in PostgreSQL
- **Multiple conversations** — create or switch between chats for a signed-in user
- **Google authentication** — Streamlit user sessions and thread ownership checks protect conversation access
- **Web search** — DuckDuckGo supplies current external information when needed
- **Calculator tool** — supports addition, subtraction, multiplication, and division
- **Short-term summarization** — older messages are summarized as conversations grow
- **Long-term memory** — durable facts and preferences are stored per user in PostgreSQL
- **Docker support** — run the application with the included Python 3.11 image

## Architecture

```text
Streamlit UI
    |
    +-- Google authentication
    +-- PDF upload and conversation management
    |
    v
LangGraph workflow
    |
    +-- remember --> chat node <---> tools
    |                      |
    |                      +-- DuckDuckGo search
    |                      +-- Calculator
    |                      +-- Thread-scoped RAG
    |
    +-- summarize

PostgreSQL
    +-- LangGraph checkpoints
    +-- Long-term user memory
    +-- User and thread ownership
    +-- PDF metadata
    +-- pgvector document chunks
```

The graph starts with a memory-extraction step, then sends the request to the Gemini chat node. The model can invoke a web search, calculator, or RAG tool. Tool results are passed back to the chat node for a final response. Conversations over ten messages trigger summarization and remove older active messages.

## Tech Stack

| Area | Technology |
|---|---|
| UI | Streamlit |
| Agent framework | LangGraph, LangChain |
| LLM | Google Gemini (`gemini-3.5-flash-lite`) |
| Embeddings | Google Gemini Embeddings (`gemini-embedding-2`) |
| Database | PostgreSQL |
| Vector search | pgvector through LangChain `PGVector` |
| PDF processing | PyPDFLoader, RecursiveCharacterTextSplitter |
| Search | DuckDuckGo Search |
| Authentication | Streamlit authentication |
| Container | Docker, Python 3.11 |

## Prerequisites

- Python 3.11+
- A Google Gemini API key
- A PostgreSQL database
- Permission to enable the PostgreSQL `vector` extension
- Streamlit authentication configuration for Google sign-in

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/meer140/langgraph-chatbot.git
cd langgraph-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Enable pgvector

Connect to your PostgreSQL database with a role that can create extensions:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The application creates its required application and LangGraph tables automatically.

### 5. Configure environment variables

Create `.env` in the project root:

```dotenv
GOOGLE_API_KEY=your_google_gemini_api_key
DB_URI=postgresql://username:password@host:5432/database_name
```

The URI must use `postgresql://`, `postgres://`, or `postgresql+psycopg://`. The application converts supported URIs for the PGVector integration.

### 6. Configure Streamlit authentication

Add Streamlit authentication settings under `.streamlit/secrets.toml`. The frontend uses `st.user`, `st.login()`, and `st.logout()`; configure Google authentication according to your Streamlit deployment environment.

A local example structure is:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "replace-with-a-random-secret"

[auth.google]
client_id = "your-google-o-auth-client-id"
client_secret = "your-google-o-auth-client-secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

### 7. Start the app

```bash
streamlit run frontend.py
```

Open the local URL printed by Streamlit, then sign in to create and manage conversations.

## Docker

The included Dockerfile uses Python 3.11 and exposes Streamlit on port `8501`.

```bash
docker build -t langgraph-chatbot .

docker run --rm -p 8501:8501 \
  -e PORT=8501 \
  -e GOOGLE_API_KEY="your_google_gemini_api_key" \
  -e DB_URI="postgresql://username:password@host:5432/database_name" \
  langgraph-chatbot
```

For deployment environments that mount `/etc/secrets/secrets.toml`, the image copies that file into `.streamlit/secrets.toml` before starting Streamlit.

## Using the Application

1. Sign in with Google.
2. Create a new chat or open a previous conversation.
3. Upload one or more PDF files from the sidebar.
4. Select **Process PDFs** to extract, split, embed, and store document chunks.
5. Ask questions in the chat box.

The assistant may use:

- `rag_tool` for questions about the active conversation's PDFs
- DuckDuckGo search for current external information
- `calculator` for arithmetic
- Gemini knowledge for general questions

## Project Structure

```text
.
├── backend.py       # LangGraph workflow, tools, RAG, PostgreSQL, and memory
├── frontend.py      # Streamlit UI, authentication, uploads, and chat streaming
├── requirements.txt # Python dependencies
├── Dockerfile       # Container build and Streamlit start command
├── pipeline.svg     # LangGraph workflow diagram
├── .dockerignore
└── .gitignore
```

## Data and Security Notes

- Chat checkpoints, user memories, PDF metadata, and pgvector embeddings are stored in PostgreSQL.
- PDF retrieval is filtered by the active LangGraph `thread_id`.
- Thread access is validated against `app_user_threads` before loading or answering in a conversation.
- Long-term memories are intentionally limited to stable facts and preferences explicitly stated by the user.
- Keep API keys, database credentials, OAuth secrets, and Streamlit cookies out of source control.

## Workflow Diagram

![LangGraph workflow pipeline](pipeline.svg)

## Live Demo

https://langgraph-chatbot-u5li.onrender.com/
