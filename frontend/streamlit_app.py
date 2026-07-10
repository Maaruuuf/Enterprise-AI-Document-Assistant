"""
Streamlit frontend for the Enterprise AI Document Assistant.

HF Spaces (Streamlit SDK) deployment note:
-------------------------------------------
HF Spaces' Streamlit SDK can only launch a single `streamlit run` process —
it cannot also run a separate FastAPI/uvicorn server (that requires the
Docker SDK). So instead of calling the API over HTTP, this version imports
the service layer (app.services.rag_pipeline, app.services.vector_store)
directly and calls the same functions in-process.

Nothing in app/services, app/models, or app/prompts changed to make this
work — only this file did. The FastAPI layer (app/main.py, app/api/routes.py)
is still in the repo and still fully functional locally via:
    uvicorn app.main:app --reload

Run this file with:
    streamlit run frontend/streamlit_app.py
"""

import sys
import uuid
from pathlib import Path

import streamlit as st

# --- Make `app.*` importable regardless of the working directory Streamlit
# was launched from (HF Spaces runs `streamlit run frontend/streamlit_app.py`
# from the repo root, but this guards against other launch setups too). ---
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services import rag_pipeline, vector_store  # noqa: E402

st.set_page_config(
    page_title="Enterprise AI Document Assistant",
    page_icon="📄",
    layout="centered",
)

# --- Custom styling ---
st.markdown("""
<style>
    .source-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-left: 4px solid #4A90D9;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .confidence-high { color: #2ea043; font-weight: 600; }
    .confidence-medium { color: #d4a72c; font-weight: 600; }
    .confidence-low { color: #e5534b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- Session state initialization ---
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None


def _create_new_thread() -> str:
    thread_id = str(uuid.uuid4())
    st.session_state.threads[thread_id] = {
        "backend_session_id": None,
        "title": None,
        "messages": [],
    }
    st.session_state.active_thread_id = thread_id
    return thread_id


def _get_active_thread() -> dict:
    if (
        st.session_state.active_thread_id is None
        or st.session_state.active_thread_id not in st.session_state.threads
    ):
        _create_new_thread()
    return st.session_state.threads[st.session_state.active_thread_id]


def _confidence_class(confidence: float) -> str:
    if confidence >= 0.75:
        return "confidence-high"
    if confidence >= 0.5:
        return "confidence-medium"
    return "confidence-low"


def run_query(question: str, backend_session_id: str | None) -> dict:
    """Call the RAG pipeline directly (in-process) instead of over HTTP.

    Mirrors the shape of the old API response so the rest of the UI code
    below needs no changes.

    Raises:
        rag_pipeline.RAGPipelineError: on infrastructure failures (embedding,
            vector search, or LLM generation failing) — caught by the caller.
    """
    response = rag_pipeline.answer_question(
        question=question,
        session_id=backend_session_id,
    )
    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "sources": [s.model_dump() for s in response.sources],
        "session_id": response.session_id,
        "session_title": response.session_title,
        "llm_model_used": response.llm_model_used,
    }


def check_backend_health() -> dict | None:
    """Equivalent of the old /health endpoint, called directly."""
    try:
        stats = vector_store.get_index_stats()
        connected = stats["total_vector_count"] >= 0
        return {
            "status": "healthy" if connected else "degraded",
            "pinecone_connected": connected,
            "documents_indexed": stats["total_vector_count"],
        }
    except Exception:
        return None


# --- Sidebar ---
with st.sidebar:
    st.title("📄 Document Assistant")
    st.caption("Enterprise AI Document Assistant")

    st.divider()

    health = check_backend_health()
    if health:
        status_icon = "🟢" if health["status"] == "healthy" else "🟡"
        st.markdown(f"{status_icon} **Status:** {health['status'].capitalize()}")
        st.markdown(f"📚 **Indexed chunks:** {health['documents_indexed']}")
    else:
        st.markdown("🔴 **Status:** Vector store unreachable")

    st.divider()

    if st.button("➕ New conversation", use_container_width=True):
        current = st.session_state.threads.get(st.session_state.active_thread_id)
        if current is None or len(current["messages"]) > 0:
            _create_new_thread()
        st.rerun()

    st.divider()
    st.markdown("**Chat history**")

    if not st.session_state.threads:
        st.caption("No conversations yet.")
    else:
        for thread_id in reversed(list(st.session_state.threads.keys())):
            thread = st.session_state.threads[thread_id]
            label = thread["title"] or "New conversation"
            is_active = thread_id == st.session_state.active_thread_id

            col_select, col_delete = st.columns([5, 1])

            with col_select:
                if st.button(
                    f"{'💬 ' if is_active else '　 '}{label}",
                    key=f"thread_btn_{thread_id}",
                    use_container_width=True,
                ):
                    st.session_state.active_thread_id = thread_id
                    st.rerun()

            with col_delete:
                if st.button("🗑️", key=f"delete_btn_{thread_id}", help="Delete this conversation"):
                    del st.session_state.threads[thread_id]
                    if st.session_state.active_thread_id == thread_id:
                        remaining = list(st.session_state.threads.keys())
                        if remaining:
                            st.session_state.active_thread_id = remaining[-1]
                        else:
                            _create_new_thread()
                    st.rerun()

    st.divider()
    st.caption(
        "Ask questions about company HR policy, leave entitlements, and "
        "employee handbook guidelines. Answers are grounded strictly in "
        "the indexed documents — if information isn't available, the "
        "assistant will say so rather than guessing."
    )

# --- Main chat area ---
active_thread = _get_active_thread()

st.title("Enterprise AI Document Assistant")
st.caption("Ask a question about company policy — I'll answer using the indexed documents only.")

for msg in active_thread["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            conf = msg.get("confidence", 0.0)
            conf_class = _confidence_class(conf)
            st.markdown(
                f"<span class='{conf_class}'>Confidence: {conf * 100:.1f}%</span>",
                unsafe_allow_html=True,
            )
            for src in msg["sources"]:
                pages_str = ", ".join(str(p) for p in src["pages"])
                st.markdown(
                    f"<div class='source-card'>📄 <b>{src['document']}</b> — page(s) {pages_str}</div>",
                    unsafe_allow_html=True,
                )

user_question = st.chat_input("Ask a question about company policy...")

if user_question:
    active_thread["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = run_query(user_question, active_thread["backend_session_id"])

                active_thread["backend_session_id"] = result["session_id"]
                if active_thread["title"] is None:
                    active_thread["title"] = result.get("session_title")

                st.markdown(result["answer"])

                conf = result["confidence"]
                conf_class = _confidence_class(conf)
                st.markdown(
                    f"<span class='{conf_class}'>Confidence: {conf * 100:.1f}%</span>",
                    unsafe_allow_html=True,
                )

                for src in result["sources"]:
                    pages_str = ", ".join(str(p) for p in src["pages"])
                    st.markdown(
                        f"<div class='source-card'>📄 <b>{src['document']}</b> — page(s) {pages_str}</div>",
                        unsafe_allow_html=True,
                    )

                active_thread["messages"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "confidence": conf,
                })

                st.rerun()

            except rag_pipeline.RAGPipelineError as e:
                error_msg = "The assistant is temporarily unable to process this request. Please try again shortly."
                st.error(error_msg)
                active_thread["messages"].append({"role": "assistant", "content": error_msg})

            except Exception:
                error_msg = "An unexpected error occurred."
                st.error(error_msg)
                active_thread["messages"].append({"role": "assistant", "content": error_msg})