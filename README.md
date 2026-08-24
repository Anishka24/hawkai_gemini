🦅 HawkAI

Welcome to HawkAI — an intelligent Legal RAG assistant for regulatory rulebooks.

The project uses Generative AI and Retrieval-Augmented Generation (RAG) to answer questions from the PRA, Basel 3.1, and HKMA rulebooks. It retrieves relevant PDF passages first, then generates a source-grounded answer with citations.

🔗 https://hawkaigemini-vfkgttda9isajihuwij6ac.streamlit.app/
🌐 HawkAI Web App: Launch HawkAI
💻 GitHub Repository: View Source Code
🤖 Legal RAG Assistant

HawkAI allows users to select a regulatory rulebook and ask questions such as:

What is PRA?
What are CET1 capital requirements?
What are the regulatory responsibilities of banks?

It searches the selected PDF and provides an answer using only relevant retrieved passages.

📚 Rulebooks Used
PRA Rulebook
Basel 3.1 Rulebook
HKMA Rulebook
🧠 Methodology

HawkAI follows the RAG pipeline:

PDF Rulebooks
→ Text extraction
→ Text chunking
→ Local embeddings
→ FAISS vector database
→ Relevant passage retrieval
→ Gemini-generated answer with citations
How it works
Regulatory PDF text is extracted page by page.
The text is split into small passages called chunks.
A local embedding model converts every chunk into numerical vectors.
FAISS stores these vectors for fast semantic search.
When a user asks a question, HawkAI retrieves the most relevant passages.
Gemini generates a clear answer only from those retrieved passages.
The answer includes PDF source and page citations.
🛠️ Tech Stack
Frontend: Streamlit
Backend: Python
PDF Processing: PyPDF
Local Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
Vector Database: FAISS
Generative AI Model: Gemini 3.6 Flash
Deployment: Streamlit Community Cloud
Version Control: GitHub
📁 Project Structure
HawkAI/
├── app.py                    # Main Streamlit application
├── build_index.py            # Creates FAISS indexes from PDFs
├── requirements.txt          # Python dependencies
├── PRA.pdf                   # PRA regulatory rulebook
├── Basel3.1.pdf              # Basel 3.1 rulebook
├── HKMA.pdf                  # HKMA rulebook
├── indexes/                  # Saved FAISS indexes and text chunks
│   ├── PRA_Rulebook.faiss
│   ├── PRA_Rulebook_chunks.json
│   ├── Basel_3_1_Rulebook.faiss
│   ├── Basel_3_1_Rulebook_chunks.json
│   ├── HKMA_Rulebook.faiss
│   └── HKMA_Rulebook_chunks.json
├── .streamlit/
│   └── secrets.toml          # Gemini API key; do not upload to GitHub
└── README.md
🚀 Deployment

The application is deployed using Streamlit Community Cloud.

Push the code, PDFs, and indexes folder to GitHub.
Open Streamlit Community Cloud.
Select the GitHub repository and app.py.
Add the Gemini API key in Advanced settings → Secrets.
Deploy the application.



🙋‍♀️ Author
Anishka Singh
🔗 GitHub*:

```toml
GEMINI_API_KEY = "your-new-key"
```
