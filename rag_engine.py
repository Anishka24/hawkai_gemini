"""RAGWise core: upload processing, Chroma search, Gemini answers, and feedback."""
import csv, os, tempfile
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = os.getenv("RAG_DATA_DIR", os.path.join(tempfile.gettempdir(), "ragwise_data"))
CHROMA_DB_DIR, UPLOAD_DIR, FEEDBACK_LOG = (os.path.join(DATA_DIR, x) for x in ("chroma_db", "uploaded_docs", "feedback_log.csv"))
EMBEDDING_MODEL_NAME, LLM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2", "gemini-3.5-flash-lite"
DEFAULT_RETRIEVAL_K, DEFAULT_RELEVANCE_THRESHOLD = 6, .5
ANSWER_PROMPT = """You are a helpful assistant answering only from the supplied document excerpts. If the excerpts do not contain the answer, say that you cannot answer based on the provided documents. Do not guess.

Conversation history:\n{history}\n\nDocument excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"""
CONDENSE_PROMPT = """Rewrite the follow-up question into a standalone question using the conversation history. Return only the rewritten question.

History:\n{history}\n\nFollow-up question: {question}\nStandalone question:"""

def get_llm(): return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=.35)
def get_embeddings(): return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
def load_existing_vectorstore(embeddings):
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings) if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR) else None

def process_uploaded_files(uploaded_files, embeddings, existing_store=None):
    os.makedirs(UPLOAD_DIR, exist_ok=True); messages, documents = [], []
    for upload in uploaded_files:
        ext = os.path.splitext(upload.name)[1].lower()
        if ext not in {".pdf", ".txt"}:
            messages.append(("warning", f"Skipped unsupported file type: {ext}")); continue
        path = os.path.join(UPLOAD_DIR, upload.name)
        with open(path, "wb") as f: f.write(upload.getbuffer())
        try:
            docs = (PyPDFLoader(path) if ext == ".pdf" else TextLoader(path)).load()
            for doc in docs: doc.metadata["source"] = upload.name
            documents.extend(docs); messages.append(("success", f"Loaded {len(docs)} page(s) from {upload.name}"))
        except Exception as error: messages.append(("error", f"Could not process {upload.name}: {error}"))
    if not documents: return existing_store, messages
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    if existing_store:
        existing_store.add_documents(chunks); messages.append(("success", f"Added {len(chunks)} chunks to the knowledge base.")); return existing_store, messages
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
    messages.append(("success", f"Created knowledge base with {len(chunks)} chunks.")); return store, messages

def get_kb_stats(vectorstore):
    if not vectorstore: return {"total_chunks":0,"total_documents":0,"by_document":{}}
    data = vectorstore.get(include=["metadatas"]).get("metadatas") or []; by_document = {}
    for meta in data:
        source = (meta or {}).get("source", "unknown"); by_document[source] = by_document.get(source,0)+1
    return {"total_chunks":len(data),"total_documents":len(by_document),"by_document":by_document}

def list_documents(vectorstore):
    return [{"name":name,"chunks":count,"path":path if os.path.exists(path) else None} for name,count in get_kb_stats(vectorstore)["by_document"].items() for path in [os.path.join(UPLOAD_DIR,name)]]

def remove_document(vectorstore, filename):
    try:
        vectorstore._collection.delete(where={"source":filename}); path=os.path.join(UPLOAD_DIR,filename)
        if os.path.exists(path): os.remove(path)
        return True, None
    except Exception as error: return False, str(error)

def _history(history):
    return "\n".join(f"{'User' if row['role']=='user' else 'Assistant'}: {row['content']}" for row in history[-4:]) or "(no previous turns)"
def retrieve_relevant(vectorstore, query, k=DEFAULT_RETRIEVAL_K, threshold=DEFAULT_RELEVANCE_THRESHOLD):
    scored=[(doc,1/(1+max(distance,0))) for doc,distance in vectorstore.similarity_search_with_score(query,k=k)]
    return [(doc,score) for doc,score in scored if score>=threshold] or scored[:1]
def confidence_label(score): return ("High","high") if score>=.75 else (("Medium","medium") if score>=.5 else ("Low","low"))

def response_text(response):
    """Return readable Gemini text across LangChain response formats.

    Recent Gemini integrations can return content as a list of blocks such
    as ``[{"type": "text", "text": "..."}]``. Rendering that list
    directly caused the raw object shown in the old interface.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text") or block.get("content")
            else:
                value = getattr(block, "text", None) or getattr(block, "content", None)
            if isinstance(value, str):
                parts.append(value)
        if parts:
            return "\n".join(parts).strip()
    return str(content).strip()

def answer_question(llm, vectorstore, history, question, k=DEFAULT_RETRIEVAL_K, threshold=DEFAULT_RELEVANCE_THRESHOLD):
    standalone=question
    if history:
        try: standalone=response_text(llm.invoke(CONDENSE_PROMPT.format(history=_history(history),question=question))) or question
        except Exception: pass
    relevant=retrieve_relevant(vectorstore,standalone,k,threshold)
    if not relevant: return {"answer":"I couldn't find anything relevant in the knowledge base.","sources":[],"confidence_label":"Low","confidence_class":"low"}
    answer=response_text(llm.invoke(ANSWER_PROMPT.format(history=_history(history),context="\n\n".join(doc.page_content for doc,_ in relevant),question=question)))
    label,css_class=confidence_label(max(score for _,score in relevant))
    return {"answer":answer,"sources":relevant,"confidence_label":label,"confidence_class":css_class}

def log_feedback(question, answer, rating):
    try:
        os.makedirs(DATA_DIR,exist_ok=True); exists=os.path.exists(FEEDBACK_LOG)
        with open(FEEDBACK_LOG,"a",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            if not exists: writer.writerow(["timestamp","question","answer_snippet","rating"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"),question,answer[:200],rating])
        return True
    except Exception: return False
def get_feedback_stats():
    if not os.path.exists(FEEDBACK_LOG): return {"total":0,"positive":0,"positive_pct":None}
    try:
        with open(FEEDBACK_LOG,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        positive=sum(row.get("rating")=="up" for row in rows)
        return {"total":len(rows),"positive":positive,"positive_pct":round(100*positive/len(rows)) if rows else None}
    except Exception: return {"total":0,"positive":0,"positive_pct":None}
