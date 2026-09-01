import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_dev_chatbot_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    save_chat_message,
    create_thread,
    update_thread_title,
    get_thread_titles,
    delete_thread,
    get_thread_titles
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


# def reset_chat():
#     thread_id = generate_thread_id()
#     st.session_state["thread_id"] = thread_id
#     add_thread(thread_id)
#     st.session_state["message_history"] = []
#     st.session_state['message_history'].append(
#         {'role': 'assistant', 'content': "Hello! How can I help you today?"})

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id

    st.session_state["message_history"] = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        }
    ]


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

threads = list(dict.fromkeys(
    str(t) for t in st.session_state["chat_threads"]
))[::-1]

current_thread = str(st.session_state["thread_id"])

if current_thread not in threads:
    threads.insert(0, current_thread)


thread_titles = get_thread_titles()
selected_thread = None

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(
            f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed",
                              state="complete", expanded=False)


# ============================ HISTORY ============================

st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")

else:
    for thread_id in threads:

        thread_id_str = str(thread_id)

        title = thread_titles.get(
            thread_id_str,
            "New thread"
        )

        # Chat row
        col1, col2 = st.sidebar.columns([5, 1])

        with col1:
            if st.button(
                title,
                key=f"side-thread-{thread_id_str}",
                use_container_width=True,
            ):
                selected_thread = thread_id

        with col2:
            if st.button(
                "🗑️",
                key=f"delete-thread-{thread_id_str}",
                help="Delete this chat",
            ):
                st.session_state["delete_confirm"] = thread_id_str
                st.rerun()

        # ---------------------------------------------------------
        # Confirmation directly BELOW THIS CHAT
        # ---------------------------------------------------------

        if st.session_state.get("delete_confirm") == thread_id_str:
            with st.sidebar.container(border=True):

                st.warning(
                    "Delete this chat permanently?",
                    icon="⚠️",
                )

                confirm_col1, confirm_col2 = st.columns(2)

                with confirm_col1:
                    if st.button(
                        "Delete",
                        key=f"confirm-delete-{thread_id_str}",
                        type="primary",
                        use_container_width=True,
                    ):
                        delete_thread(thread_id_str)

                        st.session_state["chat_threads"] = [
                            thread
                            for thread in st.session_state["chat_threads"]
                            if str(thread) != thread_id_str
                        ]

                        st.session_state["delete_confirm"] = None

                        if str(st.session_state["thread_id"]) == thread_id_str:
                            reset_chat()

                        st.rerun()

                with confirm_col2:
                    if st.button(
                        "Cancel",
                        key=f"cancel-delete-{thread_id_str}",
                        use_container_width=True,
                    ):
                        st.session_state["delete_confirm"] = None
                        st.rerun()

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    thread_key = str(st.session_state["thread_id"])

    if thread_key not in st.session_state["chat_titles"]:
        create_thread(thread_key, user_input[:30])
        st.session_state["chat_threads"].append(thread_key)
        st.session_state["chat_titles"][thread_key] = user_input[:30]

    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    save_chat_message(
        thread_id=thread_key,
        role="user",
        content=user_input,
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,  # type: ignore
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(  # type: ignore
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(message_chunk, AIMessage):
                    if isinstance(message_chunk.content, str):
                        yield message_chunk.content
                    elif isinstance(message_chunk.content, list):
                        for block in message_chunk.content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                            ):
                                yield block["text"]
        ai_message = st.write_stream(ai_only_stream())
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    if ai_message and isinstance(ai_message, str) and ai_message.strip():
        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": ai_message
            }
        )

        save_chat_message(
            thread_id=thread_key,
            role="assistant",
            content=ai_message,
        )

    if thread_key not in st.session_state["chat_titles"]:
        title = user_input[:30]
        st.session_state["chat_titles"][thread_key] = title
        update_thread_title(
            thread_id=thread_key,
            title=title,
        )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

# st.divider()

if selected_thread:

    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)
    temp_messages = []
    for msg in messages:
        # User message
        if isinstance(msg, HumanMessage):
            temp_messages.append(
                {
                    "role": "user",
                    "content": str(msg.content)
                }
            )
        # Assistant message
        elif isinstance(msg, AIMessage):

            content = ""

            if isinstance(msg.content, str):
                content = msg.content

            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, str):
                        content += block

                    elif isinstance(block, dict) and "text" in block:
                        content += block["text"]

            else:
                content = str(msg.content)

            # empty assistant messages ignore karo
            if content.strip():
                temp_messages.append(
                    {
                        "role": "assistant",
                        "content": content
                    }
                )
        # ToolMessage ignore
        elif isinstance(msg, ToolMessage):
            continue

    st.session_state["message_history"] = temp_messages

    st.session_state["ingested_docs"].setdefault(
        str(selected_thread),
        {}
    )

    st.rerun()
# First message => save title
if (user_input is not None and st.session_state["thread_id"] not in st.session_state["chat_titles"]
    ):
    st.session_state["chat_titles"][st.session_state["thread_id"]
                                    ] = user_input[:30]
    st.rerun()
