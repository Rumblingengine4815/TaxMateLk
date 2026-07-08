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
        n_results=3  # TOKEN EFFICIENCY: Only grabbing the top 3 chunks instead of 5 or 10.
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
    pass


def run_agent(question: str, user_profile: dict = None) -> str:
    """
    The main function that ties the 4 steps together.
    """
    pass


if __name__ == "__main__":
    print("Agent pipeline skeleton initialized.")
