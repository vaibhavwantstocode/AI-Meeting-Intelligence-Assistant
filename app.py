import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.pipeline import run_pipeline
from core.rag_engine import ask_question


st.set_page_config(
    page_title="AI Video Assistant",
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --muted: #9090b8;
    --success: #10b981;
    --danger: #ef4444;
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.25rem;
    background: linear-gradient(135deg, #ffffff, #9f67ff, var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 0.78rem;
}
.metric-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    padding: 1rem;
    min-height: 100%;
}
.stage-row {
    display: flex;
    gap: 0.65rem;
    align-items: center;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface-2);
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.45rem;
    font-size: 0.82rem;
}
.stage-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: var(--border);
    flex: 0 0 auto;
}
.stage-active { background: var(--accent-2); }
.stage-done { background: var(--success); }
.stage-failed { background: var(--danger); }
</style>
""",
    unsafe_allow_html=True,
)


PIPELINE_STAGES = [
    ("audio", "Audio processing"),
    ("transcript", "Transcription"),
    ("title", "Title generation"),
    ("summary", "Summarization"),
    ("extract", "Information extraction"),
    ("rag", "RAG engine"),
]


def init_state() -> None:
    defaults = {
        "result": None,
        "chat_history": [],
        "pipeline_done": False,
        "pipeline_steps": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def stage_class(stage: str) -> str:
    state = st.session_state.pipeline_steps.get(stage, "pending")
    if state == "active":
        return "stage-active"
    if state == "done":
        return "stage-done"
    if state == "failed":
        return "stage-failed"
    return ""


def render_stage(stage: str, label: str) -> None:
    st.markdown(
        f"""
        <div class="stage-row">
            <span class="stage-dot {stage_class(stage)}"></span>
            <span>{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_card(title: str, body: str) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.markdown(body or "Unavailable")


init_state()

with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.7rem">AI Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.divider()

    source = st.text_input(
        "YouTube URL or file path",
        placeholder="https://youtube.com/watch?v=... or C:\\path\\file.mp4",
    )
    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("Analyse", use_container_width=True)

    if st.session_state.pipeline_steps:
        st.divider()
        st.caption("Pipeline status")
        for stage, label in PIPELINE_STAGES:
            render_stage(stage, label)

st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Transcribe, summarize, extract, and chat with meeting videos</div>',
    unsafe_allow_html=True,
)
st.divider()

if run_btn:
    st.session_state.pipeline_done = False
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.pipeline_steps = {}

    progress_placeholder = st.empty()

    def update_step(stage: str, state: str) -> None:
        st.session_state.pipeline_steps[stage] = state

    try:
        progress_placeholder.info("Pipeline running. This can take a while for long videos.")
        st.session_state.result = run_pipeline(source, language, callback=update_step)
        st.session_state.pipeline_done = True

        if st.session_state.result.get("errors"):
            progress_placeholder.warning("Analysis completed with warnings.")
        else:
            progress_placeholder.success("Analysis complete.")
        time.sleep(0.5)
        progress_placeholder.empty()
        st.rerun()
    except Exception as exc:
        for stage, state in list(st.session_state.pipeline_steps.items()):
            if state == "active":
                st.session_state.pipeline_steps[stage] = "failed"
        progress_placeholder.error(f"Pipeline failed: {exc}")

result = st.session_state.result

if not result:
    st.info("Enter a YouTube URL or local media path in the sidebar to start.")
    st.stop()

if result.get("errors"):
    with st.expander("Pipeline warnings", expanded=True):
        for stage, error in result["errors"].items():
            st.warning(f"{stage}: {error}")

st.header(result["title"])

summary_col, transcript_col = st.columns([3, 2], gap="large")
with summary_col:
    render_text_card("Summary", result["summary"])
with transcript_col:
    with st.expander("Full transcript", expanded=False):
        st.text_area("Transcript", result["transcript"], height=300, label_visibility="collapsed")

action_col, decision_col, question_col = st.columns(3, gap="medium")
with action_col:
    render_text_card("Action Items", result["action_items"])
with decision_col:
    render_text_card("Key Decisions", result["key_decisions"])
with question_col:
    render_text_card("Open Questions", result["open_questions"])

st.divider()
st.subheader("Chat with your meeting")

rag_chain = result.get("rag_chain")
if rag_chain is None:
    st.warning("RAG chat is unavailable because the retrieval engine could not be built.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask anything about the meeting transcript")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(rag_chain, user_question)
            st.write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history and st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
