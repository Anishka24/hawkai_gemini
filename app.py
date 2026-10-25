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
