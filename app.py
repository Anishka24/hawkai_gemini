import json
import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="HawkAI", page_icon="🦅", layout="wide")

PDFS = {
    "PRA Rulebook": "PRA.pdf",
    "Basel 3.1 Rulebook": "Basel3.1.pdf",
    "HKMA Rulebook": "HKMA.pdf",
}
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 150
TOP_K = 6
INDEX_DIRECTORY = Path(__file__).parent / "indexes"


def get_api_key():
    """Read the Gemini key from Streamlit Secrets or the local .env environment."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        import os
        return os.getenv("GEMINI_API_KEY")


def index_paths(rulebook_name: str) -> tuple[Path, Path]:
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", rulebook_name).strip("_")
    return INDEX_DIRECTORY / f"{safe_name}.faiss", INDEX_DIRECTORY / f"{safe_name}_chunks.json"


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Download/cache a small local embedding model the first time it is used."""
    return SentenceTransformer(LOCAL_EMBEDDING_MODEL)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str) -> list[str]:
    passages = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            sentence_end = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if sentence_end > start + CHUNK_SIZE // 2:
                end = sentence_end + 1
        passage = text[start:end].strip()
        if passage:
            passages.append(passage)
        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return passages


def extract_pdf_passages(pdf_name: str) -> list[dict]:
    pdf_path = Path(__file__).parent / pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_name} is missing from the project folder.")
    reader = PdfReader(pdf_path)
    passages = []
    for page_number, page in enumerate(reader.pages, start=1):
        for text in split_text(clean_text(page.extract_text() or "")):
            passages.append({"text": text, "source_file": pdf_name, "page_number": page_number})
    if not passages:
        raise ValueError(f"No readable text was extracted from {pdf_name}. It may be a scanned PDF.")
    return passages


def local_embeddings(texts: list[str]) -> np.ndarray:
    """Generate embeddings locally. No Gemini embedding API request is made."""
    model = get_embedding_model()
    return np.asarray(
        model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def build_index(rulebook_name: str, progress_callback=None):
    """Build and save a FAISS index locally for one rulebook."""
    passages = extract_pdf_passages(PDFS[rulebook_name])
    vectors = []
    batch_size = 32
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        vectors.append(local_embeddings([item["text"] for item in batch]))
        if progress_callback:
            progress_callback(min((start + len(batch)) / len(passages), 1.0))
    vectors = np.vstack(vectors).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    INDEX_DIRECTORY.mkdir(exist_ok=True)
    index_path, chunks_path = index_paths(rulebook_name)
    faiss.write_index(index, str(index_path))
    chunks_path.write_text(json.dumps(passages, ensure_ascii=False), encoding="utf-8")
    return index, passages


@st.cache_resource(show_spinner=False)
def load_saved_index(rulebook_name: str):
    index_path, chunks_path = index_paths(rulebook_name)
    if not index_path.exists() or not chunks_path.exists():
        return None
    return faiss.read_index(str(index_path)), json.loads(chunks_path.read_text(encoding="utf-8"))


def retrieve(index, passages, question: str) -> list[tuple[dict, float]]:
    scores, positions = index.search(local_embeddings([question]), min(TOP_K, len(passages)))
    return [(passages[position], float(score)) for score, position in zip(scores[0], positions[0]) if position >= 0]


def generate_answer(api_key: str, rulebook_name: str, question: str, results: list[tuple[dict, float]]) -> str:
    context = "\n\n".join(
        f"[Source {number}: {item['source_file']}, page {item['page_number']}]\n{item['text']}"
        for number, (item, _) in enumerate(results, start=1)
    )
    prompt = f"""Answer the user's question using ONLY the supplied regulatory excerpts.

Rulebook: {rulebook_name}
Question: {question}

Requirements:
- Give a direct, well-structured answer.
- Cite every factual statement using [Source number].
- If the excerpts do not answer the question, say so clearly.
- Do not invent legal requirements or citations.

Regulatory excerpts:
{context}"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(max_output_tokens=1400),
    )
    return response.text


def main():
    st.title("🦅 HawkAI")
    st.caption("Legal RAG assistant for PRA, Basel 3.1, and HKMA rulebooks.")
    with st.sidebar:
        st.header("Rulebook")
        rulebook_name = st.selectbox("Choose a document", list(PDFS))
        st.caption("Search indexes are made locally—no Gemini embedding quota is used.")
        st.markdown("---")
        st.caption("Answers are generated only from retrieved PDF passages.")
    question = st.text_input("Ask a regulatory question", placeholder="Example: What are the CET1 capital requirements?")
    if st.button("Search and answer", type="primary", disabled=not question.strip()):
        try:
            saved_data = load_saved_index(rulebook_name)
            if saved_data:
                index, passages = saved_data
            else:
                with st.spinner("Creating the local search index. The first run downloads a small local model."):
                    progress = st.progress(0, text=f"Preparing {rulebook_name} for search...")
                    index, passages = build_index(rulebook_name, lambda value: progress.progress(value, text=f"Preparing {rulebook_name} for search..."))
                    progress.empty()
            results = retrieve(index, passages, question.strip())
            api_key = get_api_key()
            if not api_key:
                st.error("Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Secrets (or .env locally).")
                st.stop()
            with st.spinner("Writing an answer from the retrieved passages..."):
                answer = generate_answer(api_key, rulebook_name, question.strip(), results)
            st.subheader("Answer")
            st.write(answer)
            with st.expander("Retrieved passages"):
                for number, (item, score) in enumerate(results, start=1):
                    st.markdown(f"**Source {number} — {item['source_file']}, page {item['page_number']}** (match: {score:.2f})")
                    st.write(item["text"])
        except Exception as error:
            st.error(f"Unable to process the request: {error}")


if __name__ == "__main__":
    main()
