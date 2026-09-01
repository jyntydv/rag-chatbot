from __future__ import annotations
import streamlit as st
from psycopg.types.json import Jsonb
from psycopg.rows import DictRow, dict_row
from psycopg import Connection

import os
# import sqlite3
# from langgraph.checkpoint.sqlite import SqliteSaver
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------


llm = ChatGoogleGenerativeAI(
    # model="gemini-3.6-flash",
    model="gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
)
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = get_embeddings()

# llm = ChatOpenAI(model="gpt-4o-mini")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
# _THREAD_RETRIEVERS: Dict[str, Any] = {}
# _THREAD_METADATA: Dict[str, dict] = {}


# def _get_retriever(thread_id: Optional[str]):
#     """Fetch the retriever for a thread if available."""
#     if thread_id and thread_id in _THREAD_RETRIEVERS:
#         return _THREAD_RETRIEVERS[thread_id]
#     return None

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No text could be extracted from the PDF.")

        # Generate embeddings
        texts = [doc.page_content for doc in chunks]
        vectors = embeddings.embed_documents(texts)

        document_filename = filename or os.path.basename(temp_path)

        # Store chunks + embeddings in Supabase/PostgreSQL
        with pool.connection() as conn:
            with conn.cursor() as cur:

                # Remove previous copy of same file for this thread
                cur.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE thread_id = %s
                      AND filename = %s
                    """,
                    (str(thread_id), document_filename),
                )

                for doc, vector in zip(chunks, vectors):
                    cur.execute(
                        """
                        INSERT INTO document_chunks
                        (
                            thread_id,
                            filename,
                            content,
                            metadata,
                            embedding
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                        """,
                        (
                            str(thread_id),
                            document_filename,
                            doc.page_content,
                            Jsonb(doc.metadata),
                            vector,
                        ),
                    )

        return {
            "filename": document_filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun()


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}"
        f"&apikey={ALPHAVANTAGE_API_KEY}"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None
) -> dict:
    """
    Retrieve relevant information from the PDF uploaded
    in the current chat thread.
    """

    if not thread_id:
        return {
            "error": "thread_id is required.",
            "query": query,
        }

    # Generate query embedding
    query_embedding = embeddings.embed_query(query)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM match_document_chunks(
                    %s::extensions.vector,
                    %s,
                    %s
                )
                """,
                (
                    query_embedding,
                    str(thread_id),
                    4,
                ),
            )

            rows = cur.fetchall()

    if not rows:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    context = []
    metadata = []

    for row in rows:
        context.append(row["content"])
        metadata.append(row["metadata"])

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": rows[0]["filename"],
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
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
    response = llm_with_tools.invoke(messages, config=config)  # type:ignore
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
# conn = sqlite3.connect(database="data-rag/chatbot_rag.db",
#                        check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
    conninfo=DATABASE_URL,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    },
    min_size=1,
    max_size=5,
)
checkpointer = PostgresSaver(pool)

checkpointer.setup()

# Create LangGraph checkpoint tables if they don't exist
checkpointer.setup()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------


# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         cfg = getattr(checkpoint, "config", {}) or {}
#         configurable = cfg.get("configurable") if isinstance(
#             cfg, dict) else None
#         if isinstance(configurable, dict):
#             thread_id = configurable.get("thread_id")
#             if thread_id is not None:
#                 all_threads.add(thread_id)
#     return list(all_threads)

def save_chat_message(thread_id: str, role: str, content: str):
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (thread_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (str(thread_id), role, content),
            )


def create_thread(thread_id: str, title: str = "New thread"):
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO threads (thread_id, title)
                VALUES (%s, %s)
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (str(thread_id), title),
            )


def update_thread_title(thread_id: str, title: str):
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE threads
                SET title = %s,
                    updated_at = NOW()
                WHERE thread_id = %s
                """,
                (title, str(thread_id)),
            )


def retrieve_all_threads():
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id
                FROM threads
                ORDER BY updated_at DESC
                """
            )
            return [str(row["thread_id"]) for row in cur.fetchall()]


def get_thread_titles():
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, title
                FROM threads
                ORDER BY updated_at DESC
                """
            )
            return {
                str(row["thread_id"]): row["title"]
                for row in cur.fetchall()
            }


# def thread_has_document(thread_id: str) -> bool:
#     return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    filename,
                    COUNT(*) AS chunks,
                    COUNT(DISTINCT metadata->>'page') AS documents
                FROM document_chunks
                WHERE thread_id = %s
                GROUP BY filename
                ORDER BY MAX(created_at) DESC
                LIMIT 1
                """,
                (str(thread_id),),
            )

            row = cur.fetchone()

    if not row:
        return {}

    return {
        "filename": row["filename"],
        "chunks": row["chunks"],
        "documents": row["documents"],
    }


def delete_thread(thread_id: str):
    thread_id = str(thread_id)

    with pool.connection() as conn:
        with conn.cursor() as cur:

            # Delete readable chat messages
            cur.execute(
                """
                DELETE FROM chat_messages
                WHERE thread_id = %s
                """,
                (thread_id,),
            )

            # Delete uploaded PDF chunks / embeddings
            cur.execute(
                """
                DELETE FROM document_chunks
                WHERE thread_id = %s
                """,
                (thread_id,),
            )

            # Delete thread record
            cur.execute(
                """
                DELETE FROM threads
                WHERE thread_id = %s
                """,
                (thread_id,),
            )
