"""
RAG Evaluation Pipeline - Streamlit ui (ChatGPT-style)
Run from project root: streamlit run ui/app.py

Chat sessions are stored only in memory (session_state). No database.
"""

import io
import sys
import uuid
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# Ensure project root is on path when running from ui/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.retrieval.retriever import Retriever
from src.vector_database.chroma_db import ChromaDatabase
from src.embedding_manager.embedding_manager import EmbeddingManager
from src.pipeline.rag_pipeline import RagPipeline

# Session state keys
SESSIONS = "chat_sessions"
CURRENT_SESSION = "current_session_id"
PIPELINE = "pipeline"


def _run_silent(fn, *args, **kwargs):
    """Run a callable with stdout/stderr suppressed (so pipeline print statements don't show in ui)."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        return fn(*args, **kwargs)


def get_pipeline():
    """Create and cache the RAG pipeline (avoids re-init on every run). No extra loader—only 'Thinking...' shows when user sends a message."""
    if PIPELINE not in st.session_state:
        def _init():
            vector_store = ChromaDatabase(collection_name="research-papers")
            embedding_manager = EmbeddingManager()
            retriever = Retriever(
                vector_store=vector_store,
                embedding_manager=embedding_manager,
            )
            st.session_state[PIPELINE] = RagPipeline(retriever=retriever)
        _run_silent(_init)
    return st.session_state[PIPELINE]


def init_session_state():
    """Initialize in-memory chat sessions (no database)."""
    if SESSIONS not in st.session_state:
        st.session_state[SESSIONS] = {}  # session_id -> list of {"role": "user"|"assistant", "content": str}
    if CURRENT_SESSION not in st.session_state:
        st.session_state[CURRENT_SESSION] = None


def new_session():
    """Create a new chat session and make it current."""
    sid = str(uuid.uuid4())[:8]
    st.session_state[SESSIONS][sid] = []
    st.session_state[CURRENT_SESSION] = sid
    return sid


def get_current_messages():
    """Messages for the current session, or empty list."""
    sid = st.session_state.get(CURRENT_SESSION)
    if sid is None:
        return None
    return st.session_state[SESSIONS].get(sid, [])


def session_title(session_id: str) -> str:
    """Label for sidebar: first user message or 'New chat'."""
    messages = st.session_state[SESSIONS].get(session_id, [])
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = (m["content"] or "").strip()
            return (text[:40] + "…") if len(text) > 40 else text or "New chat"
    return "New chat"


def delete_all_sessions():
    """Clear all chat sessions from memory (temporary storage only)."""
    st.session_state[SESSIONS] = {}
    st.session_state[CURRENT_SESSION] = None


def main():
    st.set_page_config(
        page_title="RAG Evaluation Pipeline",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Increase font size + breathing space between avatar and text in chat messages
    st.markdown(
        """
        <style>
        body, .stApp, [data-testid="stAppViewContainer"], p, span, div, label, .stMarkdown {
            font-size: 21px !important;
        }
        .stChatMessage { font-size: 21px !important; }
        input, textarea { font-size: 21px !important; }
        /* More space between chat avatar (logo) and message text */
        [data-testid="stChatMessage"] > div > div:last-child,
        .stChatMessage > div > div:last-child {
            margin-left: 1.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Left sidebar: chat sessions ----
    with st.sidebar:
        st.header("Chat sessions")
        st.caption("Stored in memory only. Gone when you close the app.")
        st.divider()

        if st.button("➕ New chat", use_container_width=True):
            new_session()
            st.rerun()

        # Ensure we have at least one session when opening
        if st.session_state[CURRENT_SESSION] is None and not st.session_state[SESSIONS]:
            new_session()
            st.rerun()

        # Only list sessions that have at least one message (avoids duplicate "New chat" entry)
        session_ids = [
            sid for sid in st.session_state[SESSIONS]
            if len(st.session_state[SESSIONS].get(sid, [])) > 0
        ]
        current_sid = st.session_state.get(CURRENT_SESSION)

        for sid in session_ids:
            label = session_title(sid)
            if sid == current_sid:
                st.button(
                    f"💬 {label}",
                    key=f"sel_{sid}",
                    use_container_width=True,
                    type="primary",
                    disabled=True,
                )
            else:
                if st.button(f"💬 {label}", key=f"sel_{sid}", use_container_width=True):
                    st.session_state[CURRENT_SESSION] = sid
                    st.rerun()

        st.divider()
        if st.button("🗑️ Delete all chats", use_container_width=True, type="secondary"):
            delete_all_sessions()
            st.rerun()

    # ---- Main area: current chat ----
    st.title("🔍 RAG Evaluation Pipeline")
    st.caption("Ask a question; answers use your indexed documents. Chat input is at the bottom.")

    messages = get_current_messages()
    if messages is not None:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        # No session selected (shouldn't happen after init)
        st.info("Create or select a chat from the sidebar.")

    # ---- Chat input at bottom (Streamlit places it at bottom) ----
    if prompt := st.chat_input("Ask something..."):
        # Ensure we have a current session
        if st.session_state[CURRENT_SESSION] is None:
            new_session()

        sid = st.session_state[CURRENT_SESSION]
        st.session_state[SESSIONS][sid].append({"role": "user", "content": prompt})

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    pipeline = get_pipeline()
                    answer = _run_silent(pipeline.execute, user_query=prompt)
                    st.session_state[SESSIONS][sid].append({"role": "assistant", "content": answer})
                    st.markdown(answer)
                except Exception as e:
                    err_msg = f"Error: {e}"
                    st.session_state[SESSIONS][sid].append({"role": "assistant", "content": err_msg})
                    st.error(err_msg)
                    st.exception(e)

        st.rerun()


if __name__ == "__main__":
    main()
