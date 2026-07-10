"""
Streamlit frontend for the Enterprise AI Document Assistant.

Supports multiple chat threads (sessions) in a single browser session,
similar to ChatGPT-style sidebar history. Each thread maps 1:1 to a backend
session_id, so conversation memory (handled server-side) stays correct when
switching between threads.

Run with:
    streamlit run frontend/streamlit_app.py
"""

import os
import uuid

import requests
import streamlit as st

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
REQUEST_TIMEOUT_SECONDS = 30

st.set_page_config(
    page_title="Enterprise AI Document Assistant",
    page_icon="📄",
    layout="centered",
)

# --- Custom styling ---
# Uses Streamlit's built-in CSS variables (var(--...)) instead of hardcoded
# colors, so source cards and confidence text stay readable in both light
# and dark themes automatically.
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

    .thread-button-active {
        background-color: var(--secondary-background-color);
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session state initialization ---
# `threads`: dict mapping local_thread_id -> {backend_session_id, title, messages}
# `active_thread_id`: which thread is currently displayed/being chatted in.
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None


def _create_new_thread() -> str:
    """Create a new empty chat thread and make it active.

    Returns:
        The local thread_id of the newly created thread.
    """
    thread_id = str(uuid.uuid4())
    st.session_state.threads[thread_id] = {
        "backend_session_id": None,  # assigned after the first successful query
        "title": None,               # set from the backend's auto-generated title
        "messages": [],
    }
    st.session_state.active_thread_id = thread_id
    return thread_id


def _get_active_thread() -> dict:
    """Return the currently active thread, creating one if none exists yet."""
    if st.session_state.active_thread_id is None or st.session_state.active_thread_id not in st.session_state.threads:
        _create_new_thread()
    return st.session_state.threads[st.session_state.active_thread_id]


def _confidence_class(confidence: float) -> str:
    """Map a confidence score to a CSS class for color-coded display."""
    if confidence >= 0.75:
        return "confidence-high"
    if confidence >= 0.5:
        return "confidence-medium"
    return "confidence-low"


def call_query_api(question: str, backend_session_id: str | None) -> dict:
    """Call the backend /query endpoint.

    Args:
        question: The user's question text.
        backend_session_id: The backend session_id for this thread, or None
            if this is the thread's first message.

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.RequestException: On network failure, timeout, or non-2xx response.
    """
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question, "session_id": backend_session_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def check_backend_health() -> dict | None:
    """Check backend health for the sidebar status indicator.

    Returns:
        Health response dict, or None if the backend is unreachable.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
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
        st.markdown("🔴 **Status:** Backend unreachable")
        st.caption(f"Could not reach {API_BASE_URL}")

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

# Render conversation history for the active thread
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

# Chat input
user_question = st.chat_input("Ask a question about company policy...")

if user_question:
    active_thread["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = call_query_api(user_question, active_thread["backend_session_id"])

                # Persist backend session_id and title onto this specific thread
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

                # Rerun so the sidebar thread list updates with the new title immediately
                st.rerun()

            except requests.exceptions.Timeout:
                error_msg = "The request timed out. Please try again."
                st.error(error_msg)
                active_thread["messages"].append({"role": "assistant", "content": error_msg})

            except requests.exceptions.ConnectionError:
                error_msg = f"Could not connect to the backend at {API_BASE_URL}. Is the API running?"
                st.error(error_msg)
                active_thread["messages"].append({"role": "assistant", "content": error_msg})

            except requests.exceptions.HTTPError:
                error_msg = "The assistant encountered an error processing this question. Please try again."
                st.error(error_msg)
                active_thread["messages"].append({"role": "assistant", "content": error_msg})

            except Exception:
                error_msg = "An unexpected error occurred."
                st.error(error_msg)
                active_thread["messages"].append({"role": "assistant", "content": error_msg})