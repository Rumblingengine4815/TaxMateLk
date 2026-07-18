# TaxMate LK Deployment Instructions

## What this project needs

- Python 3.11+
- `GROQ_API_KEY`
- Optional: `HF_TOKEN`
- Optional: `FINE_TUNED_MODEL_ID`
- The ChromaDB index already present in `chroma_db/`

## Local run

### With uv

```bash
uv sync
uv run python app.py
```

### With venv

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## What to expect

- Open the app at `http://127.0.0.1:7860/`
- Save a profile in the left panel
- Ask a tax question in the chat
- Optional PDF upload works from the chat panel

## Hugging Face Spaces

1. Push the repo to a public or private Space.
2. Keep `app.py` at the repository root.
3. Keep `requirements.txt` at the repository root.
4. Add `GROQ_API_KEY` in Space secrets.
5. Add `HF_TOKEN` if you want the fine-tuned model to load from Hugging Face.
6. Add `FINE_TUNED_MODEL_ID=Lorion4815/taxmate-lk-merged` if you want the fine-tuned generation path active.
7. Start the Space and confirm the chat responds.

## Notes

- `pyproject.toml` is for local development and `uv`.
- `requirements.txt` is still needed for Spaces-style installs.
- If the fine-tuned model is unavailable, the app falls back to Groq so the demo still runs.

