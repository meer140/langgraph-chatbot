from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.documents import Document

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_postgres.vectorstores import PGVector

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from dotenv import load_dotenv

import os
import tempfile
import uuid


# ============================================================
# 1. ENVIRONMENT AND MODELS
# ============================================================

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)


# ============================================================
# 2. POSTGRESQL CONNECTION
# ============================================================

DB_URI = os.getenv("DB_URI")

if not DB_URI:
    raise ValueError("DB_URI is missing from your .env file.")


# Connection pool for LangGraph checkpoints and application tables
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=10,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
    },
)

checkPointer = PostgresSaver(pool)

# Initialize LangGraph checkpoint tables
checkPointer.setup()


# ============================================================
# 3. APPLICATION DATABASE TABLES
# ============================================================

with pool.connection() as conn:
    with conn.cursor() as cur:

        # User and thread ownership
        cur.execute("""
            CREATE TABLE IF NOT EXISTS app_user_threads (
                user_id TEXT NOT NULL,
                thread_id TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # PDF metadata and ingestion status
        cur.execute("""
            CREATE TABLE IF NOT EXISTS app_thread_documents (
                thread_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                documents INTEGER NOT NULL,
                chunks INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'processing',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)


# ============================================================
# 4. PGVECTOR CONNECTION
# ============================================================

def get_pgvector_uri(db_uri: str) -> str:
    """
    Convert the standard PostgreSQL URI into a psycopg3
    SQLAlchemy connection URI required by LangChain PGVector.
    """

    if db_uri.startswith("postgresql+psycopg://"):
        return db_uri

    if db_uri.startswith("postgresql://"):
        return db_uri.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    if db_uri.startswith("postgres://"):
        return db_uri.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    raise ValueError(
        "Unsupported DB_URI format. Use postgresql:// or "
        "postgresql+psycopg://."
    )


PGVECTOR_URI = get_pgvector_uri(DB_URI)


# Persistent vector store
vector_store = PGVector(
    embeddings=embedding_model,
    connection=PGVECTOR_URI,
    collection_name="chatbot_pdf_documents",
    use_jsonb=True,
    create_extension=False,
)


# ============================================================
# 5. PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Load a PDF, split it into chunks, and store its embeddings
    in PostgreSQL using PGVector.

    Each conversation can have one PDF.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    thread_id = str(thread_id)
    document_id = str(uuid.uuid4())
    filename = filename or "uploaded.pdf"

    temp_path = None
    chunk_ids = []
    reservation_created = False

    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("The uploaded PDF contains no readable pages.")

        # ----------------------------------------------------
        # Split PDF into chunks
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No text chunks could be extracted from the PDF.")

        # ----------------------------------------------------
        # Reserve this thread for one PDF
        # ----------------------------------------------------

        with pool.connection() as conn:
            with conn.cursor() as cur:

                cur.execute(
                    """
                    INSERT INTO app_thread_documents (
                        thread_id,
                        document_id,
                        filename,
                        documents,
                        chunks,
                        status
                    )
                    VALUES (%s, %s, %s, %s, %s, 'processing')
                    ON CONFLICT (thread_id) DO NOTHING
                    RETURNING thread_id;
                    """,
                    (
                        thread_id,
                        document_id,
                        filename,
                        len(docs),
                        len(chunks),
                    ),
                )

                reservation = cur.fetchone()

                if reservation is None:
                    raise ValueError(
                        "A PDF is already uploaded or being processed "
                        "for this conversation."
                    )

                reservation_created = True

        # ----------------------------------------------------
        # Add metadata to every chunk
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):

            chunk.metadata.update({
                "thread_id": thread_id,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
            })

            chunk_ids.append(
                f"{document_id}_{index}"
            )

        # ----------------------------------------------------
        # Generate embeddings and save in PostgreSQL
        # ----------------------------------------------------

        vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        # ----------------------------------------------------
        # Mark ingestion as completed
        # ----------------------------------------------------

        with pool.connection() as conn:
            with conn.cursor() as cur:

                cur.execute(
                    """
                    UPDATE app_thread_documents
                    SET status = 'ready'
                    WHERE thread_id = %s
                    AND document_id = %s;
                    """,
                    (thread_id, document_id),
                )

        return {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
            "thread_id": thread_id,
            "status": "success",
        }

    except Exception:

        # Remove partially inserted vectors if ingestion fails
        if chunk_ids:
            try:
                vector_store.delete(ids=chunk_ids)
            except Exception as cleanup_error:
                print(
                    "Vector cleanup failed:",
                    cleanup_error,
                )

        # Remove the processing reservation
        if reservation_created:
            try:
                with pool.connection() as conn:
                    with conn.cursor() as cur:

                        cur.execute(
                            """
                            DELETE FROM app_thread_documents
                            WHERE thread_id = %s
                            AND document_id = %s;
                            """,
                            (thread_id, document_id),
                        )

            except Exception as cleanup_error:
                print(
                    "Metadata cleanup failed:",
                    cleanup_error,
                )

        raise

    finally:

        # Remove temporary PDF file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# 6. CHECK PDF EXISTENCE
# ============================================================

def thread_has_document(thread_id: str) -> bool:

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM app_thread_documents
                    WHERE thread_id = %s
                    AND status = 'ready'
                );
                """,
                (str(thread_id),),
            )

            return cur.fetchone()[0]


# ============================================================
# 7. GET PDF METADATA
# ============================================================

def thread_document_metadata(thread_id: str) -> dict:

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    filename,
                    documents,
                    chunks,
                    created_at,
                    status
                FROM app_thread_documents
                WHERE thread_id = %s
                AND status = 'ready';
                """,
                (str(thread_id),),
            )

            result = cur.fetchone()

    if not result:
        return {}

    return {
        "filename": result[0],
        "documents": result[1],
        "chunks": result[2],
        "created_at": result[3],
        "status": result[4],
    }


# ============================================================
# 8. RAG TOOL
# ============================================================

@tool
def rag_tool(
    query: str,
    config: RunnableConfig,
) -> dict:
    """
    Retrieve relevant information from the PDF uploaded
    in the current conversation.

    The thread ID is obtained automatically from the
    LangGraph runtime configuration.
    """

    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id")

    if not thread_id:
        return {
            "error": "Thread ID is missing from runtime configuration.",
            "query": query,
        }

    if not thread_has_document(thread_id):
        return {
            "error": "No PDF is uploaded in this conversation.",
            "query": query,
        }

    # Search only the current conversation's PDF
    results = vector_store.similarity_search(
        query=query,
        k=4,
        filter={
            "thread_id": str(thread_id),
        },
    )

    if not results:
        return {
            "query": query,
            "context": [],
            "message": "No relevant information found in the PDF.",
        }

    context = []
    metadata = []

    for doc in results:

        context.append(doc.page_content)

        metadata.append({
            "filename": doc.metadata.get("filename"),
            "page": doc.metadata.get("page"),
            "chunk_index": doc.metadata.get("chunk_index"),
        })

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": metadata[0].get("filename"),
    }


# ============================================================
# 9. SEARCH TOOL
# ============================================================

search_Tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ============================================================
# 10. CALCULATOR TOOL
# ============================================================

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform basic arithmetic operations.
    Supported operations: add, sub, mul, div.
    """

    try:

        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            result = first_num / second_num

        else:
            return {
                "error": "Invalid operation. Choose add, sub, mul, or div."
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# ============================================================
# 11. BIND TOOLS
# ============================================================

tools = [
    search_Tool,
    calculator,
    rag_tool,
]

model_with_tools = model.bind_tools(tools)


# ============================================================
# 12. LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


# ============================================================
# 13. CHAT NODE
# ============================================================

def chat_node(
    state: ChatState,
    config: Optional[RunnableConfig] = None,
):
    """
    Main LLM node.

    The RAG tool receives the current thread ID through
    runtime configuration, not from the model.
    """

    existing_summary = state.get("summary", "")

    system_content = """
    You are a helpful AI assistant.

    Follow these instructions:

    1. Answer general questions using your knowledge.
    2. For questions about an uploaded PDF, use rag_tool.
    3. Never fabricate information from a PDF.
    4. If no PDF is available, ask the user to upload one.
    5. Use web search when current external information is needed.
    6. Use the calculator for arithmetic operations.
    7. Clearly distinguish information retrieved from documents
       from your general knowledge.
    """

    if existing_summary:
        system_content += f"""

        Previous conversation summary:

        {existing_summary}
        """

    system_message = SystemMessage(
        content=system_content
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = model_with_tools.invoke(
        messages,
        config=config,
    )

    return {
        "messages": [response]
    }


# ============================================================
# 14. SUMMARIZATION NODE
# ============================================================

def summarize(state: ChatState):

    existing_summary = state.get("summary", "")

    prompt = f"""
    You are a conversation memory manager for an AI assistant.

    Maintain a concise, accurate, and useful summary of the
    conversation for future interactions.

    EXISTING MEMORY:
    {existing_summary if existing_summary else "No existing memory."}

    NEW CONVERSATION:
    Review the conversation messages provided before this instruction.

    INSTRUCTIONS:

    1. Preserve important information:
       - User-provided background and preferences.
       - Goals, interests, and ongoing projects.
       - Important decisions and conclusions.
       - Technical details, tools, frameworks, and configurations.
       - Explicit constraints and requirements.

    2. Update existing memory:
       - Integrate new information without duplication.
       - Replace outdated information when corrected.
       - Give priority to the user's latest explicit statements.

    3. Maintain accuracy:
       - Never invent or assume facts.
       - Preserve exact names, numbers, dates, and technical details.
       - Distinguish confirmed facts from unresolved plans.
       - Do not treat hypothetical examples as user facts.

    4. Optimize the summary:
       - Keep it concise but informative.
       - Remove redundant explanations.
       - Prioritize information useful for future responses.
       - Do not reproduce the conversation transcript.

    OUTPUT:
    Return ONLY the updated memory summary.
    Use headings and bullet points where appropriate.
    Do not include introductory text or commentary.
    """

    message_for_summary = (
        state["messages"]
        + [HumanMessage(content=prompt)]
    )

    summary = model.invoke(
        message_for_summary
    )

    msg_to_delete = state["messages"][:-2]

    return {
        "summary": summary.content,
        "messages": [
            RemoveMessage(id=msg.id)
            for msg in msg_to_delete
        ],
    }


# ============================================================
# 15. ROUTING
# ============================================================

def route_after_chat(state: ChatState):

    last_message = state["messages"][-1]

    # Execute requested tools first
    if getattr(last_message, "tool_calls", None):
        return "tools"

    # Summarize longer conversations
    if len(state["messages"]) > 6:
        return "summarize"

    return END


# ============================================================
# 16. BUILD LANGGRAPH
# ============================================================

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("summarize", summarize)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {
        "tools": "tools",
        "summarize": "summarize",
        END: END,
    },
)

graph.add_edge("tools", "chat_node")
graph.add_edge("summarize", END)


# Compile graph with PostgreSQL checkpointing
chatbot = graph.compile(
    checkpointer=checkPointer
)


# ============================================================
# 17. THREAD MANAGEMENT
# ============================================================

def retrieve_all_threads():

    all_thread_id = set()

    for checkpoint in checkPointer.list(None):

        all_thread_id.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(all_thread_id)


def user_owns_thread(
    user_id: str,
    thread_id: str,
):

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM app_user_threads
                    WHERE user_id = %s
                    AND thread_id = %s
                );
                """,
                (user_id, thread_id),
            )

            return cur.fetchone()[0]


def create_user_thread(
    user_id: str,
    thread_id: str,
):

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO app_user_threads (
                    user_id,
                    thread_id
                )
                VALUES (%s, %s)
                ON CONFLICT (thread_id) DO NOTHING;
                """,
                (user_id, thread_id),
            )

    if not user_owns_thread(user_id, thread_id):
        raise PermissionError(
            "Thread ownership conflict."
        )


def retrieve_user_threads(user_id: str):

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT thread_id
                FROM app_user_threads
                WHERE user_id = %s
                ORDER BY created_at DESC;
                """,
                (user_id,),
            )

            threads = cur.fetchall()

    return [
        thread[0]
        for thread in threads
    ]