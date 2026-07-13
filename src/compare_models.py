"""
compare_models.py
-----------------
Evaluation script for TaxMate LK.
Runs 25 held-out test questions through two pipelines and saves a side-by-side
comparison to eval/results.csv for the final defense report.

Column A: Base LLM (Raw Groq, no RAG, no tools)
Column B: TaxMate LK Agent (RAG + Tools + our pipeline)
"""

import os
import json
import csv
from groq import Groq
from dotenv import load_dotenv
from src.agent import run_agent

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
EVAL_SET_PATH = "eval/eval_set.json"
RESULTS_PATH = "eval/results.csv"


def ask_base_llm(question: str) -> str:
    """
    Column A: Ask the raw base LLM with NO context, NO tools.
    This is the baseline that proves why RAG is necessary.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a Sri Lankan tax assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0.0,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def score_answer(answer: str, expected_keywords: list) -> int:
    """
    Simple keyword-based scoring.
    Returns the number of expected keywords found in the answer (case-insensitive).
    Max score = number of keywords in the expected_keywords list.
    """
    answer_lower = answer.lower()
    return sum(1 for kw in expected_keywords if kw.lower() in answer_lower)


def run_evaluation():
    print("=" * 60)
    print("TaxMate LK — Model Evaluation")
    print("=" * 60)

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    total_base_score = 0
    total_rag_score = 0

    for i, item in enumerate(eval_set):
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        max_score = len(expected_keywords)

        print(f"\n[{i+1}/{len(eval_set)}] Q: {question[:70]}...")

        # Column A: Base LLM
        print("  Running Base LLM...")
        base_answer = ask_base_llm(question)
        base_score = score_answer(base_answer, expected_keywords)

        # Column B: RAG Agent
        print("  Running RAG Agent...")
        rag_answer = run_agent(question)
        rag_score = score_answer(rag_answer, expected_keywords)

        total_base_score += base_score
        total_rag_score += rag_score

        results.append({
            "question": question,
            "base_llm_answer": base_answer,
            "base_llm_score": f"{base_score}/{max_score}",
            "rag_agent_answer": rag_answer,
            "rag_agent_score": f"{rag_score}/{max_score}",
        })

        print(f"  Base LLM Score: {base_score}/{max_score} | RAG Agent Score: {rag_score}/{max_score}")

    # Write to CSV
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    n = len(eval_set)
    total_keywords = sum(len(item["expected_keywords"]) for item in eval_set)
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Base LLM Total Score   : {total_base_score}/{total_keywords}")
    print(f"RAG Agent Total Score  : {total_rag_score}/{total_keywords}")
    improvement = total_rag_score - total_base_score
    print(f"Improvement from RAG   : +{improvement} keywords matched")
    print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
