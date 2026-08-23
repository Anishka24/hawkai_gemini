# HawkAI: Legal RAG with Gemini and Streamlit

HawkAI answers questions using passages retrieved from these local PDFs:

- `PRA.pdf`
- `Basel3.1.pdf`
- `HKMA.pdf`

It uses Gemini embeddings, FAISS semantic search, and Gemini 2.5 Flash for answer generation.

## Required files

Place the three PDFs in the same folder as `app.py`.

## Local run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create `.streamlit/secrets.toml` and add your new Gemini key:

   ```toml
   GEMINI_API_KEY = "your-key"
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud deployment

1. Push this folder and the three PDFs to a GitHub repository.
2. Go to `share.streamlit.io` and sign in with GitHub.
3. Select the repository, branch, and `app.py`.
4. Open **Advanced settings** > **Secrets** and add:

   ```toml
   GEMINI_API_KEY = "your-new-key"
   ```

5. Deploy.

Never place the real Gemini key in `app.py`, `requirements.txt`, or GitHub.
