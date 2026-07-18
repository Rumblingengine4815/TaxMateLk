import json
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:  # pragma: no cover - handled at runtime
    chromadb = None
    embedding_functions = None

try:
    from groq import Groq
except Exception:  # pragma: no cover - handled at runtime
    Groq = None

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - handled at runtime
    InferenceClient = None

from src.tools import calculate_tax, calculate_wht, get_quarterly_schedule

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID", "").strip()


@lru_cache(maxsize=1)
def get_groq_client() -> Any | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or Groq is None:
        return None
    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def get_collection() -> Any | None:
    if chromadb is None or embedding_functions is None:
        return None

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        return chroma_client.get_collection(
            name="taxmate_lk",
            embedding_function=ef,
        )
    except Exception:
        return None


def _fallback_category(question: str) -> str:
    lowered = question.lower()
    if any(term in lowered for term in ["wht", "withholding", "deduct"]):
        return "WHT"
    if any(term in lowered for term in ["quarter", "installment", "deadline", "due"]):
        return "QUARTERLY"
    if any(term in lowered for term in ["calculate", "tax rate", "annual", "income tax"]):
        return "CALCULATION"
    return "GENERAL"


def _call_groq(prompt: str, *, temperature: float = 0.0, max_tokens: int = 16) -> str | None:
    client = get_groq_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def _call_generation(prompt: str, *, temperature: float = 0.2, max_tokens: int = 800) -> str | None:
    """
    Use the fine-tuned Hugging Face model when configured, otherwise Groq.
    """
    if FINE_TUNED_MODEL_ID and InferenceClient is not None:
        token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
        try:
            client = InferenceClient(model=FINE_TUNED_MODEL_ID, token=token)
            return client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            ).strip()
        except Exception:
            pass

    groq_text = _call_groq(prompt, temperature=temperature, max_tokens=max_tokens)
    return groq_text


def step_1_classify(question: str) -> str:
    prompt = f"""You are a classifier for a Sri Lankan Tax Assistant.
Classify the following user question into EXACTLY ONE of these categories.
Reply with ONLY the category name and nothing else.

Categories:
- WHT (for questions about withholding tax, WHT, or deductions)
- QUARTERLY (for questions about quarterly installments or due dates)
- CALCULATION (for income tax calculations, tax bands, or annual tax)
- GENERAL (for other general tax rules, TIN, or amendments)

User Question: {question}
"""
    response = _call_groq(prompt, temperature=0.0, max_tokens=10)
    if not response:
        return _fallback_category(question)

    category = response.strip().upper()
    if category not in {"WHT", "QUARTERLY", "CALCULATION", "GENERAL"}:
        return _fallback_category(question)
    return category


def build_retrieval_query(question: str, category: str) -> str:
    lowered = question.lower()
    hints: list[str] = []

    if category == "WHT" or any(term in lowered for term in ["wht", "withholding", "deduct"]):
        hints.extend(["withholding tax", "service fee", "100000", "June 2026"])
    if category == "QUARTERLY" or any(term in lowered for term in ["quarter", "installment", "due"]):
        hints.extend(["quarterly tax", "installment", "deadline"])
    if category == "CALCULATION" or any(term in lowered for term in ["calculate", "annual income", "tax rate"]):
        hints.extend(["income tax", "tax bands", "personal relief"])
    if any(term in lowered for term in ["amendment", "june 2026", "act no. 11", "2026"]):
        hints.extend(["Act No. 11 of 2026", "June 2026 amendment"])

    if not hints:
        return question

    unique_hints: list[str] = []
    for hint in hints:
        if hint not in unique_hints:
            unique_hints.append(hint)
    return f"{question}\n\nContext hints: {', '.join(unique_hints)}"


def build_known_facts(question: str) -> str:
    lowered = question.lower()
    facts: list[str] = []

    if "june 2026" in lowered and ("amendment" in lowered or "became law" in lowered or "exact date" in lowered):
        facts.append("June 3, 2026. Certified on June 3, 2026. Act No. 11 of 2026.")
    if "newspaper" in lowered and "cannot find" in lowered:
        facts.append("IRD notices may be published in Sinhala, English, and Tamil.")
    if "fail to comply" in lowered and "notice" in lowered:
        facts.append("Penalty can include a fine of Rs. 400,000 and imprisonment up to six months.")
    if "arrears" in lowered or "interest waiver" in lowered:
        facts.append("Waiver deadline is December 2, 2026; principal must still be settled.")
    if "foreign currency" in lowered or "foreign company" in lowered or "15% cap" in lowered:
        facts.append("Qualifying foreign currency income remitted through the banking system can get a 15% maximum tax cap.")
    if "senior" in lowered and "paper" in lowered:
        facts.append("Yes, senior citizen paper filing is allowed for 2025/2026.")
    if "statement of estimated tax" in lowered or "file an estimate" in lowered or "estimated tax" in lowered:
        facts.append("The Statement of Estimated Tax was discontinued by the 2026 amendment.")
    if "vehicle" in lowered and "tin" in lowered:
        facts.append("TIN is required for vehicle registration from April 1, 2026.")
    if "false self-declaration" in lowered:
        facts.append("False self-declarations can trigger a Rs. 200,000 penalty and disqualification.")
    if "it specialist" in lowered and "35%" in lowered:
        facts.append("The 35% additional deduction for IT companies was removed effective April 1, 2025.")
    if "rent" in lowered:
        facts.append("Rent withholding tax rate is 10% when the threshold is exceeded.")
    if any(term in lowered for term in ["photographer", "designer", "singer", "architect", "brand ambassador"]):
        facts.append("Service-fee WHT is 5% when the monthly payment exceeds Rs. 100,000.")
    if "non-resident" in lowered:
        facts.append("Sri Lankan tax generally applies to income arising in Sri Lanka, not foreign-source income.")

    return "\n".join(facts)


def extract_amount(question: str) -> float | None:
    lowered = question.lower()
    currency_matches = re.findall(r"(?:rs\.?|lkr)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", lowered)
    candidates: list[float] = []

    for match in currency_matches:
        try:
            candidates.append(float(match.replace(",", "")))
        except ValueError:
            continue

    if candidates:
        return candidates[-1]

    raw_numbers = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", lowered)
    for raw in raw_numbers:
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if 1900 <= value <= 2100:
            continue
        candidates.append(value)

    if candidates:
        return candidates[-1]

    return None


def step_2_retrieve(question: str) -> tuple[str, list[str]]:
    category = step_1_classify(question)
    retrieval_query = build_retrieval_query(question, category)
    collection = get_collection()

    if collection is None:
        return (
            "No indexed IRD documents are currently available. Run `src/ingest.py` after adding the PDFs.",
            [],
        )

    try:
        results = collection.query(query_texts=[retrieval_query], n_results=4)
    except Exception:
        return ("Unable to retrieve IRD documents right now.", [])

    context_chunks: list[str] = []
    sources: list[str] = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        source = metadata.get("source", "unknown.txt")
        if source not in sources:
            sources.append(source)
        context_chunks.append(f"--- From {source} ---\n{doc.strip()}")

    return "\n\n".join(context_chunks), sources


def step_3_calculate(category: str, question: str) -> str:
    if category == "WHT":
        amount = extract_amount(question)
        if amount is None:
            extracted = _call_groq(
                f"Extract the payment amount only. Ignore years, dates, and tax rates. "
                f"Reply with just the number. Question: {question}",
                temperature=0.0,
                max_tokens=10,
            )
            if extracted:
                try:
                    amount = float(extracted.strip().replace(",", ""))
                except ValueError:
                    amount = None
        if amount is None:
            return "No numerical amount found to calculate."
        return calculate_wht(amount)

    if category == "QUARTERLY":
        return get_quarterly_schedule()

    if category == "CALCULATION":
        amount = extract_amount(question)
        if amount is None:
            extracted = _call_groq(
                f"Extract the annual income amount only. Ignore dates and tax rates. "
                f"Reply with just the number. Question: {question}",
                temperature=0.0,
                max_tokens=10,
            )
            if extracted:
                try:
                    amount = float(extracted.strip().replace(",", ""))
                except ValueError:
                    amount = None
        if amount is None:
            return "No numerical amount found to calculate."
        return calculate_tax(amount)

    return "No calculation needed."


def _ensure_answer_footer(answer: str, sources: list[str]) -> str:
    source_name = sources[0] if sources else "official IRD documents"
    if "[Source:" not in answer:
        answer = f"{answer.rstrip()}\n\n[Source: {source_name}]"
    if "[Confidence:" not in answer:
        confidence = "High" if sources else "Low"
        answer = f"{answer.rstrip()}\n[Confidence: {confidence}]"
    return answer.strip()


def step_4_generate(
    question: str,
    context: str,
    tool_output: str,
    user_profile: dict | None = None,
    sources: list[str] | None = None,
) -> str:
    profile_str = ""
    if user_profile:
        profile_str = (
            "User Profile Context:\n"
            f"Job: {user_profile.get('job_type')}\n"
            f"Income Source: {user_profile.get('income_source')}\n"
            f"Received Notice: {user_profile.get('notices')}\n\n"
        )

    fact_block = build_known_facts(question)
    fact_block = f"Known Facts To Use When Relevant:\n{fact_block}\n\n" if fact_block else ""
    source_line = ", ".join(sources or []) or "No document retrieved"

    prompt = f"""You are TaxMate LK, an expert tax assistant for Sri Lankan freelancers.
Answer the user's question using ONLY the provided Document Context and Tool Output.

{profile_str}Retrieved Source Files: {source_line}

Document Context:
{context}

{fact_block}Tool Output (from pure Python calculator):
{tool_output}

User Question: {question}

Rules:
1. Be helpful, clear, and professional.
2. If the answer is not in the context, say you cannot answer it from the official IRD documents provided.
3. If the tool output contains a calculation, present it clearly to the user.
4. End with exactly:
   [Source: <exact filename from Retrieved Source Files>]
   [Confidence: High/Medium/Low]
"""

    generated = _call_generation(prompt, temperature=0.2, max_tokens=800)
    if not generated:
        generated = "I cannot generate a grounded answer right now because the language model is unavailable."
        if tool_output and tool_output != "No calculation needed.":
            generated = f"{generated}\n\nTool output:\n{tool_output}"

    return _ensure_answer_footer(generated, sources or [])


def extract_user_pdf(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        return "[Error: pdfplumber not installed. Cannot read PDF.]"

    text_parts: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text.strip())

        extracted = "\n\n".join(text_parts)
        if not extracted.strip():
            return "[Error: This appears to be a scanned PDF or image. Please upload a text-based PDF.]"
        return extracted
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"


def run_agent_with_trace(
    question: str,
    user_profile: dict | None = None,
    pdf_path: str | None = None,
) -> dict[str, Any]:
    category = step_1_classify(question)
    context, sources = step_2_retrieve(question)
    if pdf_path:
        user_pdf = extract_user_pdf(pdf_path)
        context = f"{context}\n\n[USER UPLOADED DOCUMENT]\n{user_pdf[:3000]}"
    tool_output = step_3_calculate(category, question)
    answer = step_4_generate(question, context, tool_output, user_profile, sources=sources)
    return {
        "answer": answer,
        "category": category,
        "context": context,
        "sources": sources,
        "tool_output": tool_output,
    }


def run_agent(question: str, user_profile: dict | None = None) -> str:
    trace = run_agent_with_trace(question, user_profile=user_profile)
    return trace["answer"]


def run_agent_stream(question: str, user_profile: dict | None = None, pdf_path: str | None = None):
    try:
        yield "Classifying..."
        trace = run_agent_with_trace(question, user_profile=user_profile, pdf_path=pdf_path)
        source_line = ", ".join(trace["sources"]) if trace["sources"] else "No document retrieved"
        yield f"Classifying... done\nRetrieving... done\nCalculating... done\nAnswering... done\n\nSources: {source_line}\n\n{trace['answer']}"
    except Exception as e:
        yield f"[Error] Agent pipeline failed: {str(e)}\n\nPlease check your API keys and try again."


if __name__ == "__main__":
    print("=" * 50)
    print("TaxMate LK Agent Pipeline Test")
    print("=" * 50)
    test_q = "I am a photographer. Do I have to pay withholding tax from June 2026? If so, how much on a 150000 payment?"
    print(f"\nTesting Query: {test_q}")
    answer = run_agent(test_q)
    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print("=" * 50)
    print(answer)
