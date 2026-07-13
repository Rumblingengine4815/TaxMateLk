# TaxMate LK — Project Report
### Certified AI Engineer Professional Program — Project 03 Capstone
**Student ID:** 0742950760 | **Date:** July 2026

---

## 1. Problem Definition & Validation

### The Problem
Sri Lankan freelancers, remote workers, and self-employed professionals face a severe knowledge gap when it comes to their tax obligations. This gap became critically worse after the **Inland Revenue (Amendment) Act No. 11 of 2026**, certified on **June 3, 2026**, which introduced sweeping changes including:

- A new 5% Withholding Tax (WHT) on 30+ new professions (photographers, writers, IT specialists, musicians, etc.) for monthly payments exceeding Rs. 100,000
- Abolition of the Statement of Estimated Tax (SET)
- Mandatory TIN requirements for bank accounts and vehicle registration from April 1, 2026
- A new 15% capital gains tax on investment assets
- A December 2, 2026 deadline for an interest waiver on old tax arrears

**The validated problem:** The base LLM (Llama 3) does not know about the June 2026 amendment — it was not in the pre-training data. A freelancer asking any AI assistant about the new WHT rules will get **wrong or incomplete answers**. This project exists to fix that.

### Why AI is the Right Approach
A static FAQ website would go stale. A human accountant costs money per consultation. An AI agent that:
1. Has been fine-tuned on the actual IRD documents
2. Retrieves official source text at query time (RAG)
3. Calculates exact tax figures programmatically (tool calling)

...is the only approach that provides **accurate, cited, always-up-to-date** answers at zero marginal cost per query.

### Target User
Freelancers and remote workers in Sri Lanka — photographers, IT specialists, musicians, writers, coaches, and other self-employed professionals who receive payments from local or foreign clients and are now subject to new withholding tax rules they likely don't know about.

### Success Criteria (defined before building)
1. The agent correctly answers questions about the June 2026 amendment that the base model cannot.
2. Every answer cites an official IRD source document.
3. WHT calculations are mathematically exact.
4. A non-technical user can operate the interface without training.

---

## 2. System Architecture & Technical Depth

### Overview

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│              TaxMate LK Agent Pipeline           │
│                                                  │
│  Step 1: CLASSIFY (Groq llama-3.1-8b-instant)   │
│    → WHT / QUARTERLY / CALCULATION / GENERAL    │
│                           │                     │
│  Step 2: RETRIEVE (ChromaDB + MiniLM-L6-v2)     │
│    → Top 3 chunks from 15 IRD PDF documents     │
│                           │                     │
│  Step 3: CALCULATE (Pure Python Tools)           │
│    → calculate_wht(), calculate_tax(),           │
│       get_quarterly_schedule()                  │
│                           │                     │
│  Step 4: GENERATE (Groq + Fine-tuned context)   │
│    → Cited, structured final answer             │
└──────────────────────────────────────────────────┘
    │
    ▼
Gradio ChatInterface (with PDF upload support)
```

### Components & Decisions

#### Knowledge Base (RAG)
- **15 official IRD documents** were converted from PDF to text using `pdfplumber`, chunked, and embedded with `sentence-transformers/all-MiniLM-L6-v2`.
- Stored in a local **ChromaDB** vector database (`PersistentClient`).
- At query time, the top **3 most relevant chunks** are retrieved (deliberately limited for token efficiency — enough context, minimal cost).

**Why RAG?** The base LLM has no knowledge of post-training events (June 2026). RAG grounds the answer in the actual source text and ensures citations are real, not hallucinated.

#### QLoRA Fine-Tuning
- **Base model:** `meta-llama/Llama-3-8b-Instruct` (8B parameters)
- **Method:** QLoRA with Unsloth (2x faster than vanilla PEFT, compatible with free Colab T4 GPU)
- **Training data:** 100 Q&A pairs in Llama 3 Instruct chat format, all covering post-June 2026 Sri Lankan tax law
- **LoRA config:** r=16, alpha=16, targeting all attention + MLP projection layers
- **Result:** Published to HuggingFace at `Lorion4815/taxmate-lk-llama3`

**Why fine-tune?** RAG alone retrieves the right paragraph but does not guarantee the model correctly interprets niche Sri Lankan legal language. Fine-tuning teaches the model the specific answer format, citation style, and core facts so it does not hallucinate around retrieved context.

#### Tool Calling (Pure Python)
Three deterministic tools in `src/tools.py`:
- `calculate_wht(amount)` — Computes exact 5% WHT, handles Rs. 100,000 threshold
- `calculate_tax(annual_income)` — Applies Sri Lanka's progressive income tax slabs
- `get_quarterly_schedule()` — Returns exact installment due dates (Aug 15, Nov 15, Feb 15, May 15)

**Why tools?** LLMs are unreliable at arithmetic. A Python function gives mathematically exact results every time.

#### Model Choices
| Component | Model | Reason |
|---|---|---|
| Classification | `llama-3.1-8b-instant` (Groq) | Ultra-fast, near-zero cost, only outputs one word |
| Number extraction | `llama-3.1-8b-instant` (Groq) | Only outputs a number |
| Final generation | `llama-3.1-8b-instant` (Groq) | Free tier, 800 token output, reliable |
| Embeddings | `all-MiniLM-L6-v2` | Runs locally, no API cost, fast |

---

## 3. Implementation Quality & Robustness

### Code Organization
```
src/
├── agent.py          # 4-step pipeline (classify, retrieve, calculate, generate)
├── app.py            # Gradio UI (ChatInterface + PDF upload + user profile)
├── tools.py          # Pure Python tax calculators
├── ingest.py         # One-time PDF to ChromaDB ingestion script
└── compare_models.py # Evaluation harness (Base LLM vs RAG Agent)

eval/
└── eval_set.json     # 25 held-out evaluation questions with expected keywords

questions.txt         # 100 QLoRA training Q&A pairs (never used in eval)
```

### Robustness Features
- **Empty retrieval:** Falls back to "I cannot answer this based on the official IRD documents provided."
- **Tool errors:** try/except around all number extraction; if parsing fails, returns a graceful message
- **API failure:** If any Groq call fails, caught and reported to the Gradio UI
- **PDF upload:** pdfplumber with explicit error handling for non-text PDFs
- **Evaluation data isolation:** The 25 eval questions are completely separate from the 100 training pairs — no data leakage

---

## 4. Innovation & Problem-Solution Fit

**Novel problem:** TaxMate LK is specifically designed for the post-June 2026 Sri Lankan tax amendment. No existing AI system has this knowledge — it was enacted after all major LLM training cutoffs.

**Genuine solution fit:** The system directly addresses the real-world need of freelancers who received unexpected WHT deductions from clients in July 2026 and had no accurate resource to explain why. TaxMate LK gives them:
1. An explanation grounded in the actual law
2. An exact calculation of the amount
3. The source document so they can verify it themselves

---

## 5. Evaluation

### Evaluation Harness (`src/compare_models.py`)
A keyword-based evaluation harness compares two pipelines on 25 held-out questions:
- **Column A:** Raw base LLM (no RAG, no tools)
- **Column B:** TaxMate LK Agent (RAG + Tools)

Scoring: Each question has 3-5 expected keywords. A point is awarded for each keyword found in the answer.

### QLoRA Fine-Tuning Evaluation (Phase A vs Phase C)

| Metric | Base Model | Fine-Tuned Model |
|---|---|---|
| Questions answered correctly | 7 / 10 | 10 / 10 |
| Correct citation format | 7 / 10 | 10 / 10 |
| Net improvement | — | **+3 questions** |

**Key improvement areas:**
- Q1 (Interest waiver deadline): Base model gave wrong date → FT model correctly cited December 2, 2026
- Q4 (Capital gains rate): Base model incorrect → FT model correctly cited 15% from June 3, 2026
- Q6 (Photographer WHT): Base model gave generic answer → FT model calculated Rs. 7,500 with correct Act citation

### WandB Training Curve
Training loss decreased from **3.24** (Step 5) to **0.15** (Step 35) over 3 epochs on 100 examples.
View at: `https://wandb.ai/lorionsurjith-4815/huggingface`

### Honest Limitations
- The LoRA adapter requires a paid HF Inference Endpoint for live API serving; the production agent uses Groq with fine-tuned training patterns.
- Evaluation is keyword-based, not semantic — correct answers using synonyms may be under-scored.
- PDF upload works for text-based PDFs only; scanned IRD notices may not extract correctly.

---

## 6. User Experience & Interface

The Gradio UI (`src/app.py`) provides:
- **User Profile intake** (profession, income source) to personalize context
- **Streaming agent steps** — user sees each pipeline step in real-time
- **Collapsible evidence panel** — users can expand "View Retrieved Context & Docs" to see exact IRD text chunks
- **PDF upload** — users can upload their own IRD notice and ask questions about it
- **Example questions** sidebar for first-time users

---

## 7. Architecture Decision Log

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Vector DB | ChromaDB (local) | Pinecone | Free, no API cost, persistent |
| Embeddings | MiniLM-L6-v2 | OpenAI ada-002 | Free, local, strong performance |
| Fine-tuning method | QLoRA via Unsloth | Full fine-tune | Free Colab T4 compatible; Unsloth solved CUDA version crashes |
| Base model | Llama 3 8B Instruct | GPT-4, Gemini | Open weights, fine-tunable, Groq provides free fast inference |
| UI framework | Gradio | Streamlit | Native ChatInterface, HF Spaces native |
| PDF extraction | pdfplumber | PyMuPDF | Better text layout preservation for IRD documents |

---

*This report was prepared with AI assistance for formatting and structure. All architecture decisions, training data, evaluation results, and engineering work are original.*
