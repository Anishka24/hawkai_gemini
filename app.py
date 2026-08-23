import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader


st.set_page_config(page_title="HawkAI", page_icon="🦅", layout="wide")

PDFS = {
    "PRA Rulebook": "PRA.pdf",
    "Basel 3.1 Rulebook": "Basel3.1.pdf",
    "HKMA Rulebook": "HKMA.pdf",
}

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 180
TOP_K = 6


def get_api_key():
    """Read the key from Streamlit secrets, never from source code."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str) -> list[str]:
    """Create small, overlapping passages suitable for semantic search."""
    passages = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a sentence boundary when possible.
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
    """Extract readable text and page citations from one PDF."""
    pdf_path = Path(__file__).parent / pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_name} is missing from the repository.")

    reader = PdfReader(pdf_path)
    passages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        for text in split_text(page_text):
            passages.append(
                {
                    "text": text,
                    "source_file": pdf_name,
                    "page_number": page_number,
                }
            )

    if not passages:
        raise ValueError(
            f"No readable text was extracted from {pdf_name}. "
            "The PDF may be scanned and need OCR."
        )

    return passages


def embedding_values(client: genai.Client, contents: list[str], task_type: str) -> np.ndarray:
    """Create one Gemini embedding per item."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=contents,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([item.values for item in response.embeddings], dtype=np.float32)


@st.cache_resource(show_spinner=False)
def build_index(rulebook_name: str, api_key: str):
    """Create and cache a FAISS index once per running Streamlit app."""
    client = genai.Client(api_key=api_key)
    passages = extract_pdf_passages(PDFS[rulebook_name])

    vectors = []
    batch_size = 40
    progress = st.progress(0, text=f"Preparing {rulebook_name} for search...")

    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        batch_vectors = embedding_values(
            client,
            [item["text"] for item in batch],
            "RETRIEVAL_DOCUMENT",
        )
        vectors.append(batch_vectors)
        progress.progress(
            min((start + len(batch)) / len(passages), 1.0),
            text=f"Preparing {rulebook_name} for search...",
        )

    progress.empty()
    vectors = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return client, index, passages


def retrieve(client, index, passages, question: str) -> list[tuple[dict, float]]:
    query_vector = embedding_values(client, [question], "RETRIEVAL_QUERY")
    faiss.normalize_L2(query_vector)

    scores, positions = index.search(query_vector, min(TOP_K, len(passages)))
    return [
        (passages[position], float(score))
        for score, position in zip(scores[0], positions[0])
        if position >= 0
    ]


def generate_answer(client, rulebook_name: str, question: str, results: list[tuple[dict, float]]) -> str:
    context = "\n\n".join(
        f"[Source {number}: {item['source_file']}, page {item['page_number']}]\n{item['text']}"
        for number, (item, _) in enumerate(results, start=1)
    )

    prompt = f"""
Answer the user's question using ONLY the supplied regulatory excerpts.

Rulebook: {rulebook_name}
Question: {question}

Requirements:
- Give a direct, well-structured answer.
- Cite every factual statement using [Source number].
- If the excerpts do not answer the question, say so clearly.
- Do not invent legal requirements or citations.

Regulatory excerpts:
{context}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1400,
        ),
    )
    return response.text


def main():
    st.title("🦅 HawkAI")
    st.caption("Legal RAG assistant for PRA, Basel 3.1, and HKMA rulebooks.")

    api_key = get_api_key()
    if not api_key:
        st.error("Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Secrets.")
        st.stop()

    with st.sidebar:
        st.header("Rulebook")
        rulebook_name = st.selectbox("Choose a document", list(PDFS))
        st.caption("The first search builds a local FAISS index for the selected PDF.")
        st.markdown("---")
        st.caption("Answers are generated only from retrieved PDF passages.")

    question = st.text_input(
        "Ask a regulatory question",
        placeholder="Example: What are the CET1 capital requirements?",
    )

    if st.button("Search and answer", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Building search index and retrieving relevant passages..."):
                client, index, passages = build_index(rulebook_name, api_key)
                results = retrieve(client, index, passages, question.strip())

            with st.spinner("Generating answer with Gemini..."):
                answer = generate_answer(client, rulebook_name, question.strip(), results)

            st.subheader("Answer")
            st.markdown(answer)

            with st.expander("Retrieved source passages"):
                for number, (item, score) in enumerate(results, start=1):
                    st.markdown(
                        f"**Source {number} — {item['source_file']}, page {item['page_number']}** "
                        f"(similarity: {score:.3f})"
                    )
                    st.write(item["text"])
                    st.divider()

        except Exception as error:
            st.error(f"Unable to process the request: {error}")


if __name__ == "__main__":
    main()
