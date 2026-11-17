
import uuid
import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
)


# ==================================================
# 1. PAGE CONFIGURATION AND AUTHENTICATION
# ==================================================

st.set_page_config(
    page_title="LangGraph PDF Chatbot",
    page_icon="💬",
    layout="wide"
)

if not st.user.is_logged_in:

    st.title("LangGraph PDF Chatbot")
    st.write("Login to access your personal conversations.")

    if st.button("Login with Google"):
        st.login()

    st.stop()


user_id = st.user.sub

from backend import (
    chatbot,
    ingest_pdf,
    create_user_thread,
    retrieve_user_threads,
    user_owns_thread,
    thread_document_metadata,
    thread_has_document,
)


# ==================================================
# 2. THREAD MANAGEMENT
# ==================================================

def generate_thread_id():
    return str(uuid.uuid4())


def create_new_chat():

    thread_id = generate_thread_id()

    create_user_thread(user_id, thread_id)

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    st.rerun()


def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return "".join(text_parts)

    return ""


def load_conversation(thread_id):

    if not user_owns_thread(user_id, thread_id):
        raise PermissionError("Unauthorized conversation access.")

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    messages = state.values.get("messages", [])

    history = []

    for message in messages:

        if isinstance(message, HumanMessage):

            history.append({
                "role": "user",
                "content": extract_text(message.content)
            })

        elif isinstance(message, AIMessage):

            if not message.tool_calls and message.content:

                history.append({
                    "role": "assistant",
                    "content": extract_text(message.content)
                })

    return history


# ==================================================
# 3. INITIALIZE SESSION
# ==================================================

# Prevent one authenticated user from seeing another
# user's previous Streamlit session history.

if st.session_state.get("session_user_id") != user_id:

    thread_id = generate_thread_id()

    create_user_thread(user_id, thread_id)

    st.session_state["session_user_id"] = user_id
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    thread_id = generate_thread_id()

    create_user_thread(user_id, thread_id)

    st.session_state["thread_id"] = thread_id


# Verify current thread ownership

if not user_owns_thread(user_id, st.session_state["thread_id"]):

    thread_id = generate_thread_id()

    create_user_thread(user_id, thread_id)

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []


# Restore messages

if "message_history" not in st.session_state:

    st.session_state["message_history"] = load_conversation(
        st.session_state["thread_id"]
    )


# ==================================================
# 4. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("Chat History")

    st.caption(f"Logged in: {st.user.email}")

    if st.button("Logout", use_container_width=True):

        st.logout()
        st.stop()

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()

    st.divider()


    # --------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------

    st.subheader("Upload PDFs")

    # Initialize uploader key
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    current_thread = st.session_state["thread_id"]

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state['uploader_key']}"
    )

    if uploaded_files:

        if st.button("Process PDFs", use_container_width=True):

            if not user_owns_thread(user_id, current_thread):
                st.error("Unauthorized conversation access.")
                st.stop()

            successful = 0
            failed = 0

            for pdf in uploaded_files:

                try:
                    with st.spinner(f"Processing {pdf.name}..."):

                        result = ingest_pdf(
                            file_bytes=pdf.getvalue(),
                            thread_id=current_thread,
                            filename=pdf.name
                        )

                    successful += 1

                    st.success(
                        f"{result['filename']} processed successfully. "
                        f"Pages: {result['documents']}, "
                        f"Chunks: {result['chunks']}"
                    )

                except Exception as e:

                    failed += 1

                    st.error(
                        f"Failed to process {pdf.name}: {e}"
                    )

            st.info(
                f"Processing completed: {successful} successful, "
                f"{failed} failed."
            )

            # Reset uploader by assigning a new widget key
            st.session_state["uploader_key"] += 1

            st.rerun()


    st.divider()

    # --------------------------------------------------
    # DISPLAY UPLOADED DOCUMENTS
    # --------------------------------------------------

    st.subheader("Uploaded Documents")

    metadata = thread_document_metadata(current_thread)

    if metadata:

        for document in metadata:

            with st.expander(document["filename"]):

                st.write("Pages:", document["documents"])
                st.write("Chunks:", document["chunks"])
                st.write("Status:", document["status"])

    else:

        st.caption("No PDFs uploaded in this conversation.")

    st.divider()


    # --------------------------------------------------
    # PREVIOUS CONVERSATIONS
    # --------------------------------------------------

    st.subheader("Previous Conversations")

    chat_threads = retrieve_user_threads(user_id)

    for thread_id in chat_threads:

        if thread_id == st.session_state["thread_id"]:
            label = f"Current Chat ({thread_id[:8]})"
        else:
            label = f"Chat ({thread_id[:8]})"

        if st.button(
            label,
            key=f"thread_{thread_id}",
            use_container_width=True
        ):

            try:

                history = load_conversation(thread_id)

                st.session_state["thread_id"] = thread_id
                st.session_state["message_history"] = history

                st.rerun()

            except PermissionError:

                st.error("You cannot access this conversation.")


# ==================================================
# 5. MAIN CHAT INTERFACE
# ==================================================

st.title("LangGraph PDF Chatbot")

st.caption(
    f"Thread ID: {st.session_state['thread_id']}"
)


# Display conversation history

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================================
# 6. USER INPUT AND LANGGRAPH STREAMING
# ==================================================

user_input = st.chat_input("Ask something...")

if user_input:

    thread_id = st.session_state["thread_id"]

    if not user_owns_thread(user_id, thread_id):

        st.error("Unauthorized conversation access.")
        st.stop()


    # Display user message

    with st.chat_message("user"):

        st.markdown(user_input)


    CONFIG = {
        "configurable": {
            "thread_id": thread_id
        }
    }


    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        try:

            for message_chunk, metadata in chatbot.stream(

                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                config=CONFIG,

                stream_mode="messages"

            ):

                if not isinstance(
                    message_chunk,
                    (AIMessageChunk, AIMessage)
                ):
                    continue

                # Ignore tool calls

                if getattr(message_chunk, "tool_calls", None):
                    continue

                chunk_text = extract_text(message_chunk.content)

                if chunk_text:

                    full_response += chunk_text

                    response_placeholder.markdown(
                        full_response + "▌"
                    )


            response_placeholder.markdown(full_response)


            # Restore complete conversation from PostgreSQL

            st.session_state["message_history"] = load_conversation(
                thread_id
            )

        except Exception as e:

            st.error(f"An error occurred: {e}")
