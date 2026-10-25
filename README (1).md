# RAGWise — Personal RAG Assistant

This repository now follows the RAGWise project design. Upload PDF or TXT documents and ask questions grounded in their content.

## Features

- Upload one or more PDF/TXT files
- Extract text and split it into overlapping chunks
- Create local MiniLM embeddings and store them in ChromaDB
- Search the six most relevant document passages for each question
- Generate a Gemini answer using only retrieved passages
- Ask natural follow-up questions using recent chat history
- View source passages, relevance labels, document/chunk dashboard, and feedback metrics
- Remove individual documents or reset the full knowledge base

## Flow

`Upload → extract text → chunk → embed → ChromaDB → semantic retrieval → Gemini answer + sources`

## Run locally

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# Windows PowerShell
$env:GOOGLE_API_KEY="your-key"

# macOS/Linux
export GOOGLE_API_KEY="your-key"

streamlit run app.py
```

Open the local Streamlit link, go to **Documents**, upload files, click **Process Uploaded Documents**, and then ask questions in **Chat**.

## Data storage

The app stores uploaded documents, ChromaDB data, and feedback in the operating system temporary directory by default. Set `RAG_DATA_DIR` before starting Streamlit if you want a persistent custom location.

The earlier regulatory PDF files remain in the repository as sample documents. Upload them through the app if you want to query them.
