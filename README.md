# HawkAI: local FAISS search with Gemini answers

HawkAI searches `PRA.pdf`, `Basel3.1.pdf`, and `HKMA.pdf`.

It now creates the embeddings locally with `all-MiniLM-L6-v2`. This does not
call Gemini for embeddings and does not consume Gemini embedding quota. Gemini
is used only to write the final answer from retrieved passages.

## On your laptop

1. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Create `.env` beside `app.py` and add:

   ```text
   GEMINI_API_KEY="your-new-key"
   ```

3. Build local search indexes. The first run downloads a small local AI model:

   ```bash
   python build_index.py --rulebook "PRA Rulebook"
   python build_index.py --rulebook "Basel 3.1 Rulebook"
   python build_index.py --rulebook "HKMA Rulebook"
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

Push `app.py`, `build_index.py`, `requirements.txt`, the three PDFs, and the
generated `indexes` folder to GitHub. Do not upload `.env`.

In Streamlit Community Cloud, add this in **Advanced settings → Secrets**:

```toml
GEMINI_API_KEY = "your-new-key"
```
