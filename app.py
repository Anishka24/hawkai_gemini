"""RAGWise Streamlit interface."""
import os
import shutil
import streamlit as st
import rag_engine as rag

st.set_page_config(page_title="RAGWise", page_icon="📚", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif} h1,h2,h3{font-family:'Source Serif 4',serif}
.card{background:#fff;border:1px solid #E4E1D9;border-radius:10px;padding:1rem 1.2rem}.label{color:#6B6B6B;font-size:.78rem;text-transform:uppercase}.value{font:700 1.9rem 'Source Serif 4',serif}.badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.75rem;font-weight:600}.high{background:#E3F4EA;color:#1E8E5A}.medium{background:#FBF0DC;color:#C98A1F}.low{background:#FAE6E6;color:#C24444}.empty{border:1px dashed #E4E1D9;border-radius:10px;padding:2rem;text-align:center;color:#666;background:#F6F4EF}
</style>""", unsafe_allow_html=True)

def api_key():
    try: key = st.secrets.get("GOOGLE_API_KEY")
    except Exception: key = None
    return key or os.getenv("GOOGLE_API_KEY")

if not api_key():
    st.error("No Gemini API key found. Set `GOOGLE_API_KEY` in your environment or Streamlit secrets, then restart the app.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key()

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "embeddings" not in st.session_state: st.session_state.embeddings = rag.get_embeddings()
if "llm" not in st.session_state: st.session_state.llm = rag.get_llm()
if "current_vectorstore" not in st.session_state: st.session_state.current_vectorstore = rag.load_existing_vectorstore(st.session_state.embeddings)
if "relevance_threshold" not in st.session_state: st.session_state.relevance_threshold = rag.DEFAULT_RELEVANCE_THRESHOLD

def has_kb(): return st.session_state.current_vectorstore is not None

def sources(items):
    if not items: return
    with st.expander(f"📎 Sources ({len(items)})"):
        for n, (doc, score) in enumerate(items, 1):
            meta = doc.metadata or {}
            st.markdown(f"**{n}. {meta.get('source','unknown')} — page {meta.get('page','N/A')}**")
            st.caption(f"Semantic match: {score:.0%}")
            st.write(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
            if n < len(items): st.divider()

with st.sidebar:
    st.markdown("## 📚 RAGWise")
    st.caption("Chat with your own PDFs and text documents.")
    view = st.radio("Navigate", ["💬 Chat", "📊 Dashboard", "📁 Documents"], label_visibility="collapsed")
    st.divider()
    st.session_state.relevance_threshold = st.slider("Relevance threshold", 0.0, 1.0, st.session_state.relevance_threshold, .05, help="Raise it for stricter results; lower it if useful context is missed.")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []; st.rerun()
    if st.button("⚠️ Clear Knowledge Base", use_container_width=True):
        shutil.rmtree(rag.CHROMA_DB_DIR, ignore_errors=True); shutil.rmtree(rag.UPLOAD_DIR, ignore_errors=True)
        st.session_state.current_vectorstore = None; st.session_state.chat_history = []; st.rerun()

def chat():
    st.title("Chat")
    if not has_kb():
        st.markdown('<div class="empty"><b>No knowledge base yet.</b><br><br>Open <b>📁 Documents</b>, upload PDF/TXT files, process them, then return here.</div>', unsafe_allow_html=True); return
    for i, turn in enumerate(st.session_state.chat_history):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                if turn.get("confidence_label"): st.markdown(f'<span class="badge {turn.get("confidence_class","low")}">{turn["confidence_label"]} relevance</span>', unsafe_allow_html=True)
                sources(turn.get("sources", []))
                a, b, _ = st.columns([1,1,8])
                if a.button("👍", key=f"up_{i}"): rag.log_feedback(st.session_state.chat_history[i-1]["content"] if i else "", turn["content"], "up")
                if b.button("👎", key=f"down_{i}"): rag.log_feedback(st.session_state.chat_history[i-1]["content"] if i else "", turn["content"], "down")
    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.chat_history.append({"role":"user", "content":question})
        with st.chat_message("assistant"), st.spinner("Finding relevant passages and preparing an answer..."):
            result = rag.answer_question(st.session_state.llm, st.session_state.current_vectorstore, st.session_state.chat_history[:-1], question, threshold=st.session_state.relevance_threshold)
            st.markdown(result["answer"]); st.markdown(f'<span class="badge {result["confidence_class"]}">{result["confidence_label"]} relevance</span>', unsafe_allow_html=True); sources(result["sources"])
        st.session_state.chat_history.append({"role":"assistant", "content":result["answer"], "sources":result["sources"], "confidence_label":result["confidence_label"], "confidence_class":result["confidence_class"]})
        st.rerun()

def dashboard():
    st.title("Knowledge Base Dashboard")
    stats, feedback = rag.get_kb_stats(st.session_state.current_vectorstore), rag.get_feedback_stats()
    questions = sum(x["role"] == "user" for x in st.session_state.chat_history)
    for col, item in zip(st.columns(4), [("Documents",stats["total_documents"]),("Chunks",stats["total_chunks"]),("Questions asked",questions),("Positive feedback", f'{feedback["positive_pct"]}%' if feedback["positive_pct"] is not None else "—")]):
        col.markdown(f'<div class="card"><div class="label">{item[0]}</div><div class="value">{item[1]}</div></div>', unsafe_allow_html=True)
    if stats["by_document"]: st.subheader("Chunks per document"); st.bar_chart(stats["by_document"])
    else: st.markdown('<div class="empty">No documents have been uploaded yet.</div>', unsafe_allow_html=True)

def documents():
    st.title("Documents")
    files = st.file_uploader("Select PDF or TXT files", type=["pdf","txt"], accept_multiple_files=True)
    if st.button("Process Uploaded Documents", disabled=not files):
        with st.spinner("Extracting text, creating chunks, and indexing documents..."):
            store, messages = rag.process_uploaded_files(files, st.session_state.embeddings, st.session_state.current_vectorstore)
        st.session_state.current_vectorstore = store
        for level, message in messages: getattr(st, level)(message)
        if store: st.rerun()
    st.subheader("Knowledge base contents")
    docs = rag.list_documents(st.session_state.current_vectorstore)
    if not docs: st.markdown('<div class="empty">No documents in the knowledge base yet.</div>', unsafe_allow_html=True); return
    for doc in docs:
        with st.container(border=True):
            left,right = st.columns([5,1]); left.markdown(f"**{doc['name']}** · {doc['chunks']} chunk(s)")
            if right.button("Remove", key=f"remove_{doc['name']}"):
                ok, error = rag.remove_document(st.session_state.current_vectorstore, doc["name"])
                st.toast(f"Removed {doc['name']}") if ok else st.error(error); st.rerun()
            if doc["path"]:
                with open(doc["path"], "rb") as f: content=f.read()
                st.download_button("⬇️ Download original", content, doc["name"], key=f"download_{doc['name']}")
                if doc["name"].lower().endswith(".txt"): st.text_area("Preview", content.decode("utf-8",errors="replace")[:3000], height=180, key=f"preview_{doc['name']}")

{"💬 Chat":chat,"📊 Dashboard":dashboard,"📁 Documents":documents}[view]()
