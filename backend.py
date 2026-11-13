from langgraph.graph import StateGraph, START, END
from typing import Annotated, Any, Dict, Optional, TypedDict
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,RemoveMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import os
import tempfile

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


#PDF retriever store (per thread)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


search_Tool = DuckDuckGoSearchRun(region='us-en')


@tool
def calculator(first_num: float , second_num: float, operation: str) -> dict:

    """
    A simple calculator tool to perform basic arithmetic operations.

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
            return {"error": "Invalid operation. Please choose from 'add', 'sub', 'mul', or 'div'."}    

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}    


tools = [search_Tool,calculator,rag_tool]
model_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary : str

def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = model_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


def summarize(state: ChatState):

    existing_summary = state.get('summary', "")

    prompt = f"""
        You are a conversation memory manager for an AI assistant.

        Your task is to maintain a concise, accurate, and useful summary
        of the conversation so the assistant can respond correctly in
        future interactions.

        ### EXISTING MEMORY
        {existing_summary if existing_summary else "No existing memory available."}

        ### NEW CONVERSATION
        Review the conversation messages provided above this instruction.

        ### INSTRUCTIONS

        1. Preserve important information:
        - User identity and relevant background.
        - User preferences and communication style.
        - Goals, interests, and ongoing projects.
        - Important decisions and conclusions.
        - Technical details, tools, frameworks, and configurations.
        - Explicit constraints and requirements.
        - Relevant facts the user expects the assistant to remember.

        2. Update existing memory:
        - Integrate new information into the existing summary.
        - Do not duplicate information.
        - Update outdated information when the user provides corrections.
        - Give priority to the user's latest explicit statements.
        - Preserve important historical information when it remains relevant.

        3. Handle information carefully:
        - Never invent, assume, or infer facts that the user has not provided.
        - Distinguish confirmed information from unresolved plans.
        - Preserve exact names, numbers, dates, technical terms, and
            project details when important.
        - Do not treat hypothetical examples as facts about the user.
        - Do not preserve temporary or irrelevant conversational details.

        4. Optimize memory:
        - Keep the summary concise but sufficiently informative.
        - Remove redundant explanations and repeated information.
        - Organize related information logically.
        - Prioritize information that will improve future responses.
        - Avoid storing entire conversation transcripts.

        5. Handle corrections:
        - If the user corrects previously provided information,
            replace the incorrect information.
        - Never retain contradictory facts as equally valid.
        - Preserve explicit instructions about information that
            should not be mentioned or used.

        ### OUTPUT FORMAT

        Return ONLY the updated memory summary.

        Use clear headings and bullet points where appropriate.
        Do not include introductory text, explanations, or commentary.
"""

    message_for_summary = state["messages"] + [HumanMessage(content=prompt)]
    summary = model.invoke(message_for_summary)

    msg_to_delete = state["messages"][:-2]

    return {
        "summary": summary.content,
        "messages": [RemoveMessage(id=msg.id) for msg in msg_to_delete],
    }


def route_after_chat(state: ChatState):

    last_message = state["messages"][-1]

    # Execute tools first
    if getattr(last_message, "tool_calls", None):
        return "tools"

    # Summarize if necessary
    if len(state["messages"]) > 6:
        return "summarize"

    return END


tool_node = ToolNode(tools)

DB_URI = os.getenv("DB_URI")

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=10,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0
    }
)

checkPointer = PostgresSaver(pool)

# Run once when initializing the database
checkPointer.setup()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools",tool_node)
graph.add_node("summarize",summarize)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {
        "tools": "tools",
        "summarize": "summarize",
        END: END
    }
)

graph.add_edge("tools", "chat_node")
graph.add_edge("summarize", END)



chatbot = graph.compile(checkpointer=checkPointer)

def retrieve_all_threads():
    all_thread_id = set()
    for checkpoint in checkPointer.list(None):
        all_thread_id.add(checkpoint.config['configurable']['thread_id'])

    return list(all_thread_id)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
