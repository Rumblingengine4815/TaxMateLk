import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from src.tools import calculate_wht, get_quarterly_schedule, calculate_tax

load_dotenv()

try:
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
except KeyError:
    print("Error: GROQ_API_KEY not found in .env file.")
    exit(1)

# FIX 1: Use relative path so it works both locally AND on HF Spaces
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="taxmate_lk", embedding_function=ef)


def step_1_classify(question: str) -> str:
    prompt = f"""You are a classifier for a Sri Lankan Tax Assistant.
Classify the following user question into EXACTLY ONE of these categories. Reply with ONLY the category name.

Categories:
- WHT (Questions about Withholding Tax, 5% deduction, new professions June 2026)
- QUARTERLY (Questions about installment dates, deadlines, quarterly payments)
- CALCULATION (Questions asking to calculate total income tax)
- GENERAL (General tax rules, exemptions, TIN, SET abolition, interest waiver)

User Question: {question}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )
    category = response.choices[0].message.content.strip().upper()
    # Ensure valid category
    if not any(c in category for c in ["WHT", "QUARTERLY", "CALCULATION", "GENERAL"]):
        category = "GENERAL"
    return category


def step_2_retrieve(question: str) -> tuple[str, list[str]]:
    results = collection.query(query_texts=[question], n_results=3)
    context_chunks = []
    sources = []
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        if source not in sources:
            sources.append(source)
        context_chunks.append(f"--- From {source} ---\n{doc.strip()}")
    return "\n\n".join(context_chunks), sources


def step_3_calculate(category: str, question: str) -> str:
    if category == "WHT":
        prompt = f"Extract the numerical LKR amount from this question. Reply ONLY with the number (e.g. 150000). Question: {question}"
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        try:
            amount = float(response.choices[0].message.content.strip().replace(',', '').replace('LKR', '').strip())
            result = calculate_wht(amount)
            # FIX 2: Check threshold — WHT only applies above LKR 100,000
            import json
            data = json.loads(result)
            if amount <= 100000:
                return json.dumps({"status": "not_applicable", "message": f"WHT does NOT apply. Payment of LKR {amount:,.0f} is below the LKR 100,000 monthly threshold. [Source: Amendment Act No. 11 of 2026]"})
            return result
        except (ValueError, AttributeError):
            return "No numerical amount found to calculate."

    elif category == "QUARTERLY":
        return get_quarterly_schedule()

    elif category == "CALCULATION":
        prompt = f"Extract the annual income amount from this question. Reply ONLY with the number (e.g. 4000000). Question: {question}"
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        try:
            amount = float(response.choices[0].message.content.strip().replace(',', '').strip())
            return calculate_tax(amount)
        except (ValueError, AttributeError):
            return "No numerical amount found to calculate."

    return "No calculation needed."


def step_4_generate(question: str, context: str, tool_output: str, user_profile: dict = None) -> str:
    profile_str = ""
    if user_profile and user_profile.get("job_type"):
        profile_str = f"User Profile:\nJob: {user_profile.get('job_type')}\nIncome Source: {user_profile.get('income_source')}\n\n"

    # FIX 3: Always use Groq — don't try HF Inference API for LoRA adapter
    # The fine-tuned model's knowledge is captured via the system prompt and RAG
    # HF Inference API doesn't support LoRA adapters on free tier
    prompt = f"""You are TaxMate LK, an expert tax assistant EXCLUSIVELY for Sri Lankan freelancers.
You ONLY answer questions about Sri Lanka's Inland Revenue Department (IRD) and the June 2026 Amendment Act No. 11 of 2026.
NEVER mention IRS, US tax, UK tax, or any other country's tax system.

{profile_str}Document Context (from official IRD documents):
{context}

Tool Output (from pure Python tax calculator):
{tool_output}

User Question: {question}

RULES:
1. Answer clearly and professionally using ONLY the Document Context and Tool Output above.
2. If the tool output contains a calculation, present it clearly with the numbers.
3. If the answer is not in the context, say "I cannot find this in the official IRD documents provided. Please consult a tax professional."
4. You MUST end your response with exactly:
   [Source: <most relevant filename from context>]
   [Confidence: High/Medium/Low]
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()


def extract_user_pdf(pdf_path: str) -> str:
    """Extract text from user-uploaded PDF document."""
    try:
        import pdfplumber
    except ImportError:
        return "[Error: pdfplumber not installed.]"
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text.strip())
        extracted = "\n\n".join(text_parts)
        if not extracted.strip():
            return "[This appears to be a scanned PDF. Please upload a text-based PDF.]"
        return extracted[:3000]  # limit to 3000 chars
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"


def run_agent(question: str, user_profile: dict = None) -> str:
    """Main agent — used by compare_models.py for eval."""
    category = step_1_classify(question)
    context, _ = step_2_retrieve(question)
    tool_output = step_3_calculate(category, question)
    return step_4_generate(question, context, tool_output, user_profile)


def run_agent_stream(question: str, user_profile: dict = None, pdf_path: str = None):
    """Streaming agent — used by app.py for Gradio UI."""
    try:
        yield "🧠 **Step 1: Classifying your question...**"
        category = step_1_classify(question)

        yield f"🧠 **Step 1: Classified as `{category}`** ✅\n📚 **Step 2: Retrieving official IRD documents...**"
        context, sources = step_2_retrieve(question)

        # Inject user PDF if uploaded
        if pdf_path:
            yield f"🧠 **Step 1:** ✅  📚 **Step 2:** ✅\n📄 **Reading your uploaded document...**"
            pdf_text = extract_user_pdf(pdf_path)
            context += f"\n\n[YOUR UPLOADED DOCUMENT]\n{pdf_text}"

        sources_str = ", ".join(sources) if sources else "IRD Documents"
        context_html = f"<details><summary>📄 Documents retrieved: {sources_str}</summary>\n\n```\n{context[:1500]}...\n```\n</details>\n\n"

        yield f"🧠 **Step 1:** ✅  📚 **Step 2:** ✅  ({sources_str})\n🧮 **Step 3: Running tax calculations...**\n\n{context_html}"
        tool_output = step_3_calculate(category, question)

        yield f"🧠 **Step 1:** ✅  📚 **Step 2:** ✅  🧮 **Step 3:** ✅\n✍️ **Step 4: Generating cited answer...**\n\n{context_html}"
        final_answer = step_4_generate(question, context, tool_output, user_profile)

        yield f"{context_html}{final_answer}"

    except Exception as e:
        yield f"⚠️ **Pipeline Error:** {str(e)}\n\nPlease check your API keys and try again."


if __name__ == "__main__":
    print("=" * 50)
    print("TaxMate LK Agent Pipeline Test")
    print("=" * 50)
    test_q = "I am a photographer earning Rs 150,000/month. Do I pay WHT under the 2026 amendment?"
    print(f"\nQ: {test_q}")
    print(f"\nA: {run_agent(test_q)}")
