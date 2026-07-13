import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from src.tools import calculate_wht, get_quarterly_schedule, calculate_tax

# Load environment variables
load_dotenv()

# Initialize API clients and DB
try:
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
except KeyError:
    print("Error: GROQ_API_KEY not found in .env file.")
    exit(1)

chroma_client = chromadb.PersistentClient(path="chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="taxmate_lk", embedding_function=ef)


def step_1_classify(question: str) -> str:
    """
    Step 1: Send the question to the LLM to classify it into one of our categories
    (WHT, QUARTERLY, CALCULATION, GENERAL).
    """
    prompt = f"""You are a classifier for a Sri Lankan Tax Assistant.
Classify the following user question into EXACTLY ONE of these categories. Reply with ONLY the category name and nothing else.

Categories:
- WHT (Questions about Withholding Tax, 5% deduction, etc.)
- QUARTERLY (Questions about installment dates, deadlines, quarterly payments)
- CALCULATION (Questions asking to calculate total income tax)
- GENERAL (General tax rules, exemptions, generic advice)

User Question: {question}
"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )
    
    category = response.choices[0].message.content.strip().upper()
    return category


def step_2_retrieve(question: str) -> str:
    """
    Step 2: Take the question and retrieve the top 3 most relevant chunks from our local ChromaDB.
    (Limited to 3 chunks to ensure maximum token efficiency!)
    """
    print(f"\\n[Step 2 - RETRIEVE] Searching ChromaDB...")
    
    results = collection.query(
        query_texts=[question],
        n_results=3  # High token efficiency: Only grabbing the top 3 chunks instead of 5 or 10 for maximum use possible.
    )
    
    context_chunks = []
    sources = set()
    
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        sources.add(source)
        # We strip extra whitespace to save a few more tokens!
        context_chunks.append(f"--- From {source} ---\\n{doc.strip()}")
        
    print(f"[Step 2 - RETRIEVE] Found chunks from: {', '.join(sources)}")
    return "\\n\\n".join(context_chunks)


def step_3_calculate(category: str, question: str) -> str:
    """
    Step 3: If the category requires math, extract the number from the question 
    and call our pure Python tools in tools.py.
    """
    print(f"\\n[Step 3 - CALCULATE] Checking if math tools are needed for category: {category}")
    
    if category == "WHT":
        prompt = f"Extract the numerical amount from this question. Reply ONLY with the number (e.g. 150000). Question: {question}"
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=10 # TOKEN EFFICIENCY: We only need the number!
        )
        try:
            amount = float(response.choices[0].message.content.strip().replace(',', ''))
            print(f"[Step 3 - CALCULATE] Extracted amount: {amount}. Calling calculate_wht...")
            return calculate_wht(amount)
        except ValueError:
            return "No numerical amount found to calculate."
            
    elif category == "QUARTERLY":
        print(f"[Step 3 - CALCULATE] Calling get_quarterly_schedule...")
        return get_quarterly_schedule()
        
    elif category == "CALCULATION":
        prompt = f"Extract the annual income amount from this question. Reply ONLY with the number (e.g. 4000000). Question: {question}"
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=10 # TOKEN EFFICIENCY
        )
        try:
            amount = float(response.choices[0].message.content.strip().replace(',', ''))
            print(f"[Step 3 - CALCULATE] Extracted amount: {amount}. Calling calculate_tax...")
            return calculate_tax(amount)
        except ValueError:
            return "No numerical amount found to calculate."
            
    return "No calculation needed."


def step_4_generate(question: str, context: str, tool_output: str, user_profile: dict = None) -> str:
    """
    Step 4: Send the retrieved context, tool output, and user profile to the LLM 
    to generate the final cited answer.
    """
    print(f"\\n[Step 4 - GENERATE] Synthesizing final answer...")
    
    profile_str = ""
    if user_profile:
        profile_str = f"User Profile Context:\\nJob: {user_profile.get('job_type')}\\nIncome Source: {user_profile.get('income_source')}\\n\\n"
        
    # The model is now fully merged on HF (Lorion4815/taxmate-lk-merged)
    # We can call it via Inference API if HF_TOKEN is available.
    model_id = os.environ.get("FINE_TUNED_MODEL_ID")
    client_to_use = groq_client
    
    if model_id:
        try:
            from huggingface_hub import InferenceClient
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
            if hf_token:
                print(f"[Step 4] Using FINE-TUNED model: {model_id}")
                client_to_use = InferenceClient(token=hf_token)
            else:
                print("[Step 4] FINE_TUNED_MODEL_ID set but no HF_TOKEN. Falling back to Groq.")
                model_id = "llama-3.1-8b-instant"
        except ImportError:
            print("[Step 4] huggingface_hub not installed. Falling back to Groq.")
            model_id = "llama-3.1-8b-instant"
    else:
        print("[Step 4] Using Groq base model.")

    prompt = f"""You are TaxMate LK, an expert tax assistant for Sri Lankan freelancers.
Answer the user's question using ONLY the provided Document Context and Tool Output.

{profile_str}Document Context:
{context}

Tool Output (from pure Python calculator):
{tool_output}

User Question: {question}

RULES (Follow strictly):
1. Be helpful, clear, and professional.
2. If the answer is not in the context, say "I cannot answer this based on the official IRD documents provided."
3. If the tool output contains a calculation, present it clearly to the user.
4. You MUST end your response with exactly two tags on new lines:
   [Source: filename.txt]
   [Confidence: High/Medium/Low]
"""

    if "llama" in model_id.lower() and "instant" in model_id.lower():
        response = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    else:
        try:
            response = client_to_use.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Step 4] HF API Error: {e}. Falling back to Groq.")
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
        return "[Error: pdfplumber not installed. Cannot read PDF.]"
        
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text.strip())
        
        extracted = "\\n\\n".join(text_parts)
        if not extracted.strip():
            return "[Error: This appears to be a scanned PDF or image. Please upload a text-based PDF.]"
        return extracted
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"


def run_agent(question: str, user_profile: dict = None) -> str:
    """
    The main function that ties the 4 steps together.
    """
    category = step_1_classify(question)
    context = step_2_retrieve(question)
    tool_output = step_3_calculate(category, question)
    final_answer = step_4_generate(question, context, tool_output, user_profile)
    
    return final_answer


def run_agent_stream(question: str, user_profile: dict = None, pdf_path: str = None):
    """
    Yields intermediate progress strings to be displayed in a Gradio UI,
    followed by the final answer.
    """
    try:
        # Step 1
        yield "🧠 **Step 1: Classifying...**"
        category = step_1_classify(question)
        
        # Step 2
        yield f"🧠 **Step 1: Classifying...** ✅ (Category: {category})\\n📚 **Step 2: Retrieving official IRD documents...**"
        context = step_2_retrieve(question)
        
        # Inject user uploaded PDF if provided
        user_pdf_context = ""
        if pdf_path:
            yield f"🧠 **Step 1: Classifying...** ✅\\n📚 **Step 2: Retrieving official IRD documents...** ✅\\n📄 **Reading User Document...**"
            extracted_text = extract_user_pdf(pdf_path)
            user_pdf_context = f"\\n\\n[USER UPLOADED DOCUMENT]\\n{extracted_text[:3000]}"
            context += user_pdf_context
        
        # Format the context into a dropdown
        context_html = f"<details><summary>📄 View Retrieved Context & Docs</summary>\\n\\n```text\\n{context}\\n```\\n</details>\\n\\n"
        
        # Step 3
        yield f"🧠 **Step 1: Classifying...** ✅\\n📚 **Step 2: Retrieving official IRD documents...** ✅\\n🧮 **Step 3: Calculating tax (if applicable)...**\\n\\n{context_html}"
        tool_output = step_3_calculate(category, question)
        
        # Step 4
        yield f"🧠 **Step 1: Classifying...** ✅\\n📚 **Step 2: Retrieving official IRD documents...** ✅\\n🧮 **Step 3: Calculating tax...** ✅\\n✍️ **Step 4: Synthesizing final answer...**\\n\\n{context_html}"
        final_answer = step_4_generate(question, context, tool_output, user_profile)
        
        # Final output
        yield f"{context_html}{final_answer}"
        
    except Exception as e:
        yield f"⚠️ **Error in Agent Pipeline:** {str(e)}\\n\\nPlease check your API keys and try again."


if __name__ == "__main__":
    print("="*50)
    print("TaxMate LK Agent Pipeline Test")
    print("="*50)
    
    test_q = "I am a photographer. Do I have to pay withholding tax from June 2026? If so, how much on a 150000 payment?"
    
    print(f"\\nTesting Query: {test_q}")
    answer = run_agent(test_q)
    
    print("\\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(answer)
