"""RAGWise: a polished document-chat experience built with Streamlit."""

import os
import shutil

import streamlit as st

import rag_engine as rag

st.set_page_config(page_title="RAGWise · Document Intelligence", page_icon="✦", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root { --ink:#17201d; --muted:#68736d; --forest:#123d34; --jade:#1e8e6e; --mint:#dff3ea; --cream:#faf9f5; --line:#e4e7e3; --sun:#f4bc5a; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:radial-gradient(circle at 80% -10%, #dcefe8 0, transparent 31rem), var(--cream); }
#MainMenu, footer { visibility:hidden; }
[data-testid="stSidebar"] { background:#103a32; border-right:0; }
[data-testid="stSidebar"] * { color:#f8fffb !important; }
[data-testid="stSidebar"] .stButton button { border:1px solid rgba(255,255,255,.22); background:rgba(255,255,255,.06); }
[data-testid="stSidebar"] .stButton button:hover { background:rgba(255,255,255,.15); border-color:rgba(255,255,255,.45); }
h1,h2,h3 { font-family:'Playfair Display',serif; letter-spacing:-.03em; }
.brand { display:flex; align-items:center; gap:.7rem; margin:.4rem 0 1.9rem; font-size:1.35rem; font-weight:700; }
.brand-mark { width:34px; height:34px; display:grid; place-items:center; border-radius:11px; color:#103a32 !important; background:linear-gradient(135deg,#dff3ea,#f7ce73); font-weight:800; }
.eyebrow { color:var(--jade); font-weight:700; letter-spacing:.1em; font-size:.74rem; text-transform:uppercase; }
.hero { padding:1rem 0 1.55rem; max-width:740px; }
.hero h1 { margin:.25rem 0 .5rem; font-size:3rem; line-height:1.08; }
.hero p { color:var(--muted); font-size:1.05rem; line-height:1.65; margin:0; }
.panel { background:rgba(255,255,255,.87); border:1px solid var(--line); border-radius:18px; padding:1.15rem 1.2rem; box-shadow:0 8px 35px rgba(20,61,52,.055); }
.metric { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; min-height:95px; }
.metric-label { color:var(--muted); font-size:.73rem; font-weight:700; letter-spacing:.075em; text-transform:uppercase; }
.metric-value { font-family:'Playfair Display',serif; font-size:1.9rem; font-weight:700; margin-top:.12rem; }
.badge { display:inline-block; border-radius:99px; padding:.25rem .62rem; font-size:.72rem; font-weight:700; margin-top:.15rem; }
.high { background:#dff3ea; color:#146c50; }.medium { background:#fff1d5; color:#976414; }.low { background:#fce5e2; color:#a33e35; }
.empty { margin-top:1rem; border:1px dashed #b9cbc2; border-radius:20px; background:linear-gradient(135deg,#fff,#f3faf6); padding:3rem 2rem; text-align:center; }
.empty-icon { display:grid; place-items:center; width:58px; height:58px; border-radius:18px; margin:0 auto 1rem; background:#dff3ea; font-size:1.7rem; }
.doc-card { background:#fff; border:1px solid var(--line); border-radius:15px; padding:.9rem 1rem; margin:.55rem 0; }
.doc-title { font-weight:700; font-size:.95rem; }.doc-meta { color:var(--muted); font-size:.82rem; margin-top:.2rem; }
.stChatMessage { background:rgba(255,255,255,.78); border:1px solid var(--line); border-radius:16px; padding:.35rem .65rem; margin-bottom:.7rem; }
[data-testid="stChatInput"] { border:1px solid #b7cbc1; border-radius:16px; background:#fff; box-shadow:0 8px 30px rgba(20,61,52,.08); }
.source-head { color:var(--forest); font-size:.87rem; font-weight:700; }.source-score { color:var(--muted); font-size:.78rem; }
.quick-label { color:var(--muted); font-size:.8rem; font-weight:700; margin:.6rem 0 .3rem; }
</style>
""",
    unsafe_allow_html=True,
)


def resolve_api_key():
    try:
        secret = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        secret = None
    return secret or os.getenv("GOOGLE_API_KEY")


key = resolve_api_key()
if not key:
    st.error("Gemini key missing. Add `GOOGLE_API_KEY` in Streamlit Cloud → Settings → Secrets, then reboot the app.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = key

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = rag.get_embeddings()
if "llm" not in st.session_state:
    st.session_state.llm = rag.get_llm()
if "current_vectorstore" not in st.session_state:
    st.session_state.current_vectorstore = rag.load_existing_vectorstore(st.session_state.embeddings)
if "relevance_threshold" not in st.session_state:
    st.session_state.relevance_threshold = rag.DEFAULT_RELEVANCE_THRESHOLD


def has_knowledge_base():
    return st.session_state.current_vectorstore is not None


def display_sources(items):
    if not items:
        return
    with st.expander(f"Evidence used · {len(items)} passages"):
        for number, (document, score) in enumerate(items, 1):
            metadata = document.metadata or {}
            st.markdown(
                f'<div class="source-head">{number}. {metadata.get("source", "Unknown document")} · page {metadata.get("page", "N/A")}</div>'
                f'<div class="source-score">Semantic match · {score:.0%}</div>',
                unsafe_allow_html=True,
            )
            st.write(document.page_content[:650] + ("…" if len(document.page_content) > 650 else ""))
            if number < len(items):
                st.divider()


def answer_question(question):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Reading your documents and composing an answer…"):
            result = rag.answer_question(
                st.session_state.llm,
                st.session_state.current_vectorstore,
                st.session_state.chat_history[:-1],
                question,
                threshold=st.session_state.relevance_threshold,
            )
        st.markdown(result["answer"])
        st.markdown(
            f'<span class="badge {result["confidence_class"]}">{result["confidence_label"]} relevance</span>',
            unsafe_allow_html=True,
        )
        display_sources(result["sources"])
    st.session_state.chat_history.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"],
         "confidence_label": result["confidence_label"], "confidence_class": result["confidence_class"]}
    )


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span> RAGWise</div>', unsafe_allow_html=True)
    st.caption("Your private space for asking better questions of your documents.")
    st.divider()
    view = st.radio("Navigate", ["Upload documents", "Ask your documents", "Document overview"], label_visibility="collapsed")
    st.divider()
    st.caption("ANSWER QUALITY")
    st.session_state.relevance_threshold = st.slider(
        "Relevance threshold", 0.0, 1.0, st.session_state.relevance_threshold, 0.05,
        help="Increase for stricter evidence; reduce it if the app misses relevant text.",
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("Reset knowledge base", use_container_width=True):
        shutil.rmtree(rag.CHROMA_DB_DIR, ignore_errors=True)
        shutil.rmtree(rag.UPLOAD_DIR, ignore_errors=True)
        st.session_state.current_vectorstore = None
        st.session_state.chat_history = []
        st.rerun()


def chat_view():
    st.markdown('<div class="hero"><div class="eyebrow">Document intelligence</div><h1>Ask. Explore. Understand.</h1><p>RAGWise finds the most relevant passages in your documents, then turns them into clear, evidence-backed answers.</p></div>', unsafe_allow_html=True)
    if not has_knowledge_base():
        st.markdown('<div class="empty"><div class="empty-icon">📄</div><h2>Your library is waiting</h2><p>Add a PDF or TXT document in the Library tab to begin a focused conversation.</p></div>', unsafe_allow_html=True)
        return

    for index, turn in enumerate(st.session_state.chat_history):
        avatar = "👤" if turn["role"] == "user" else "🤖"
        with st.chat_message(turn["role"], avatar=avatar):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                if turn.get("confidence_label"):
                    st.markdown(f'<span class="badge {turn["confidence_class"]}">{turn["confidence_label"]} relevance</span>', unsafe_allow_html=True)
                display_sources(turn.get("sources", []))
                left, right, _ = st.columns([1, 1, 9])
                if left.button("Helpful", key=f"up_{index}"):
                    rag.log_feedback(st.session_state.chat_history[index - 1]["content"] if index else "", turn["content"], "up")
                    st.toast("Thanks — feedback saved.")
                if right.button("Not helpful", key=f"down_{index}"):
                    rag.log_feedback(st.session_state.chat_history[index - 1]["content"] if index else "", turn["content"], "down")
                    st.toast("Thanks — feedback saved.")

    st.markdown('<div class="quick-label">TRY A STARTER QUESTION</div>', unsafe_allow_html=True)

    if st.button("Summarize this document", use_container_width=True):
        st.session_state.quick_question = "Summarize this document"
        st.rerun()

    question = st.chat_input("Ask anything about your documents…")
    question = question or st.session_state.pop("quick_question", None)
    if question:
        answer_question(question)
        st.rerun()


def library_view():
    st.markdown('<div class="hero"><div class="eyebrow">Your knowledge base</div><h1>Build your library.</h1><p>Upload PDFs or text documents. They are chunked and indexed locally for semantic search.</p></div>', unsafe_allow_html=True)
    with st.container(border=True):
        files = st.file_uploader("Drop documents here", type=["pdf", "txt"], accept_multiple_files=True, help="PDF and TXT files are supported.")
        if st.button("Add to knowledge base", type="primary", disabled=not files):
            with st.spinner("Extracting text, indexing passages, and preparing your library…"):
                store, messages = rag.process_uploaded_files(files, st.session_state.embeddings, st.session_state.current_vectorstore)
            st.session_state.current_vectorstore = store
            for level, message in messages:
                getattr(st, level)(message)
            if store:
                st.balloons()
                st.rerun()

    documents = rag.list_documents(st.session_state.current_vectorstore)
    st.subheader("Documents in your library")
    if not documents:
        st.markdown('<div class="empty"><div class="empty-icon">✦</div><h3>No documents yet</h3><p>Upload your first PDF or TXT file above.</p></div>', unsafe_allow_html=True)
        return
    for document in documents:
        with st.container(border=True):
            name_col, action_col = st.columns([6, 1])
            name_col.markdown(f'<div class="doc-title">📄 {document["name"]}</div><div class="doc-meta">{document["chunks"]} indexed passages · ready to search</div>', unsafe_allow_html=True)
            if action_col.button("Remove", key=f"remove_{document['name']}"):
                ok, error = rag.remove_document(st.session_state.current_vectorstore, document["name"])
                st.toast(f"Removed {document['name']}") if ok else st.error(error)
                st.rerun()
            if document["path"]:
                with open(document["path"], "rb") as file:
                    st.download_button("Download original", file.read(), document["name"], key=f"download_{document['name']}")


def insights_view():
    st.markdown('<div class="hero"><div class="eyebrow">Knowledge base health</div><h1>See what your library knows.</h1><p>Track document coverage, questions asked, and answer feedback in one place.</p></div>', unsafe_allow_html=True)
    stats, feedback = rag.get_kb_stats(st.session_state.current_vectorstore), rag.get_feedback_stats()
    questions = sum(item["role"] == "user" for item in st.session_state.chat_history)
    values = [("Documents", stats["total_documents"]), ("Indexed passages", stats["total_chunks"]), ("Questions asked", questions), ("Helpful answers", f'{feedback["positive_pct"]}%' if feedback["positive_pct"] is not None else "—")]
    for column, (label, value) in zip(st.columns(4), values):
        column.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.write("")
    if stats["by_document"]:
        with st.container(border=True):
            st.subheader("Indexed passages by document")
            st.bar_chart(stats["by_document"], color="#1e8e6e")
    else:
        st.markdown('<div class="empty"><div class="empty-icon">📊</div><h3>No insights yet</h3><p>Add a document to populate your dashboard.</p></div>', unsafe_allow_html=True)


{"Chat": chat_view, "Library": library_view, "Insights": insights_view}[view]()
