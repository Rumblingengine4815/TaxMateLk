---
title: TaxMate LK
emoji: 🇱🇰
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.35.0"
app_file: app.py
pinned: false
license: mit
---

# TaxMate LK

Sri Lankan tax advisory assistant for freelancers and remote workers.

## What it does

- Answers tax questions with retrieval from official IRD documents
- Supports PDF upload for a user-provided notice or document
- Uses a fine-tuned Hugging Face model when `FINE_TUNED_MODEL_ID` is set
- Falls back to Groq if the fine-tuned model is unavailable

## Run locally

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

Open the app at:

`http://127.0.0.1:7860/`

## Required files

- `app.py` for the root launch entrypoint
- `requirements.txt` for Spaces installs
- `pyproject.toml` for local `uv` development
- `chroma_db/` for the local vector store

## Environment variables

- `GROQ_API_KEY`
- `HF_TOKEN` if you want the fine-tuned model path
- `FINE_TUNED_MODEL_ID=Lorion4815/taxmate-lk-merged`

## Hugging Face Spaces

1. Put this repo in a Space.
2. Keep `app.py` at the repo root.
3. Add `GROQ_API_KEY` in Space secrets.
4. Add `HF_TOKEN` if the fine-tuned model should load from Hugging Face.
5. Add `FINE_TUNED_MODEL_ID` if you want the fine-tuned generation path active.
6. Start the Space and verify the chat responds.

## Notes

- The app UI lives in `src/app.py`.
- The deployment instructions are in `deployment_instructions.md`.
- The run guide is in `run_app.md`.
- Demo prep notes are in `C:\Users\User\Documents\Codex\2026-07-15\updated-todo-list-paste-this-to\outputs\demo_prep.md`.

