# TaxMate LK - Project Report
### Certified AI Engineer Professional Program - Project 03 Capstone
**Student ID:** 0742950760 | **Date:** July 2026

---

## 1. Problem

Sri Lankan freelancers and remote workers need reliable guidance on tax obligations after the Inland Revenue (Amendment) Act No. 11 of 2026, certified on June 3, 2026.

Key changes covered by this project:
- 5 percent withholding tax for qualifying service payments above Rs. 100,000
- Statement of Estimated Tax changes
- TIN-related requirements
- Quarterly installment guidance
- Interest waiver and notice-related questions

The base model alone is not enough because it predates the June 2026 amendment. TaxMate LK solves this with retrieval, tools, and a fine-tuned generation path.

---

## 2. Solution

TaxMate LK uses four visible steps:

1. Classify the question
2. Retrieve the most relevant official IRD chunks from ChromaDB
3. Calculate the tax result with pure Python tools
4. Generate a cited answer, using the fine-tuned model when `FINE_TUNED_MODEL_ID` is available

This keeps the pipeline explainable and lets the grader see that the techniques work together.

---

## 3. Architecture

### Core stack
- Groq `llama-3.1-8b-instant` for classification and fallback generation
- Hugging Face fine-tuned model for final generation when configured
- ChromaDB for local retrieval
- `sentence-transformers/all-MiniLM-L6-v2` embeddings
- `pdfplumber` for PDF text extraction
- Gradio for the UI

### Main files
- [src/agent.py](src/agent.py)
- [src/app.py](src/app.py)
- [src/tools.py](src/tools.py)
- [src/compare_models.py](src/compare_models.py)
- [app.py](app.py)

---

## 4. Validation Evidence

### Offline tests
- `python test_suite.py tools`: passed
- `python test_suite.py env`: passed
- `python -c "from src.app import demo"`: passed
- `python -c "from src.agent import run_agent"`: passed

### Runtime smoke tests
- Chat smoke test on a live tax question returned a cited answer and confidence footer
- PDF smoke test on `knowledge_base_raw/11-2026_E.pdf` extracted 55,811 characters successfully
- Gradio backend smoke test via `src.app.respond()` returned a full answer for a quarterly question
- Gradio backend smoke test via `src.app.respond()` with a real PDF upload returned a cited answer and source trace

### Final eval
- Base model: `34/93`
- RAG/fine-tuned agent: `67/93`

This is the strongest current evidence that the system is working end to end.

---

## 5. Fine-Tuning

The fine-tuned model id is currently set to:

`Lorion4815/taxmate-lk-merged`

Behavior:
- If `FINE_TUNED_MODEL_ID` is set, it is used for the final generation step
- If not, the agent falls back to Groq so the demo still works

This is the right tradeoff for demo stability and Gold-style evidence.

---

## 6. What Worked

- RAG over official IRD documents works
- The tax calculation tools work
- The Gradio frontend loads cleanly
- PDF extraction works on real PDFs
- The eval score improved substantially over the base model

---

## 7. What Did Not Fully Finish

- Hugging Face Spaces deployment has not been completed yet
- Browser screenshots still need to be captured
- Demo video still needs to be recorded

---

## 8. Limitations

- Keyword-based evaluation is strict and can under-score semantically correct answers
- PDF extraction works best on text-based PDFs, not scanned image-only PDFs
- The fine-tuned model is optional at runtime and still depends on the HF setup being configured correctly

---

## 9. Deployment Status

Current status:
- Local app is working
- Root launcher exists
- Fine-tuned model wiring exists
- No Supabase is required
- Hugging Face Spaces now requires a PRO subscription for Gradio apps, so deployment has been pivoted to Render.com.

Deployment link:
- `TBD` (Pending Render deployment)

---

## 10. Screenshots To Capture

Capture these for the defense and final submission:
- Home screen with the TaxMate LK header
- Profile form filled in
- Example question selection or typed question
- Live agent step trace during a response
- Retrieved source document panel expanded
- PDF upload working with a real IRD PDF
- Final cited answer with `[Source:]` and `[Confidence:]`

---

## 11. Conclusion

TaxMate LK now has:
- a working agent pipeline,
- a Gradio frontend,
- a fine-tuned-model fallback path,
- a local retrieval store,
- and a strong eval result showing real improvement over the base model.

The remaining work is deployment and presentation polish, not core functionality.
