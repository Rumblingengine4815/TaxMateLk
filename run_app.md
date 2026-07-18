# How to Run TaxMate LK

## Local run with uv

```bash
cd "C:\Users\User\Desktop\NoobDev CAIEP AI Engineering\Project 3"
uv sync
uv run python app.py
```

## Local run with venv

```bash
cd "C:\Users\User\Desktop\NoobDev CAIEP AI Engineering\Project 3"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Open the app

- Go to `http://127.0.0.1:7860/`
- Save your profile in the left panel
- Ask a tax question in the chat
- Upload a PDF if needed

## If the app does not start

- Check that `GROQ_API_KEY` is set in `.env`
- Check that `chroma_db/` exists
- Make sure no other app is already using port `7860`

## Hugging Face Spaces

- Keep `app.py` in the repo root
- Keep `requirements.txt` in the repo root
- Add `GROQ_API_KEY` in Space secrets
- Optional: add `HF_TOKEN`
- Optional: add `FINE_TUNED_MODEL_ID=Lorion4815/taxmate-lk-merged`

