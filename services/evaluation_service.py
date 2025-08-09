import os
import google.generativeai as genai
import time
import json
from tqdm.auto import tqdm

# --- CONFIGURATION & INITIALIZATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Use a specific model for evaluation tasks
evaluation_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- CONSOLIDATED EVALUATION PROMPT ---
# This new prompt asks the LLM to evaluate all metrics in a single call
# and return a structured JSON object.
CONSOLIDATED_EVALUATION_PROMPT = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to evaluate the quality of a generated answer based on a given question and the context that was used to create it.
Please evaluate the following three metrics and provide a score from 0.0 to 1.0 for each.

METRICS:
1.  **Faithfulness**: Does the answer remain factually consistent with the provided context? It is faithful if all claims in the answer can be verified from the context. It must not add new information or contradict the context.
2.  **Answer Relevance**: Is the answer directly relevant to the user's question? It should not be vague, generic, or off-topic.
3.  **Context Precision**: Was the provided context useful for answering the question? It is precise if it contains the necessary information to directly answer the question.

QUESTION:
{question}

CONTEXT:
---
{context}
---

ANSWER:
{answer}

---
Your response MUST be a valid JSON object with the following structure:
{{
  "faithfulness": <score_float>,
  "answer_relevance": <score_float>,
  "context_precision": <score_float>
}}
"""

# --- EVALUATION FUNCTIONS ---

def evaluate_single_item(item: dict) -> dict:
    """
    Runs all evaluations for a single item from the dataset using a single, consolidated LLM call.
    """
    question = item['question']
    answer = item['generated_answer']
    context = "\n\n".join(item['retrieved_context'])

    prompt = CONSOLIDATED_EVALUATION_PROMPT.format(
        question=question,
        context=context,
        answer=answer
    )

    scores = {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_precision": 0.0
    }

    try:
        print("    Scoring all metrics with a single API call...")
        response = evaluation_model.generate_content(prompt)
        
        # Clean the response to extract only the JSON part
        json_response_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON response
        parsed_scores = json.loads(json_response_str)
        
        scores["faithfulness"] = float(parsed_scores.get("faithfulness", 0.0))
        scores["answer_relevance"] = float(parsed_scores.get("answer_relevance", 0.0))
        scores["context_precision"] = float(parsed_scores.get("context_precision", 0.0))

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Warning: Could not parse JSON scores from LLM response. Defaulting to 0. Error: {e}")
        # The scores will remain 0.0 as initialized
    
    item['scores'] = scores
    return item

def run_evaluation_pipeline(results_with_context: list) -> dict:
    """
    Runs the full evaluation pipeline on a list of RAG outputs.
    """
    print(f"\n--- Starting Evaluation Pipeline for {len(results_with_context)} items ---")
    
    detailed_results = []
    
    for item in tqdm(results_with_context, desc="Evaluating RAG performance"):
        evaluated_item = evaluate_single_item(item)
        detailed_results.append(evaluated_item)
        # Add a delay to stay well within the requests-per-minute limit
        time.sleep(4)

    # Calculate average scores
    if not detailed_results:
        return {"summary_scores": {}, "detailed_results": []}

    avg_faithfulness = sum(item['scores']['faithfulness'] for item in detailed_results) / len(detailed_results)
    avg_answer_relevance = sum(item['scores']['answer_relevance'] for item in detailed_results) / len(detailed_results)
    avg_context_precision = sum(item['scores']['context_precision'] for item in detailed_results) / len(detailed_results)

    summary = {
        "average_faithfulness": round(avg_faithfulness, 3),
        "average_answer_relevance": round(avg_answer_relevance, 3),
        "average_context_precision": round(avg_context_precision, 3)
    }

    print("\n--- Evaluation Summary ---")
    print(f"Average Faithfulness: {summary['average_faithfulness']}")
    print(f"Average Answer Relevance: {summary['average_answer_relevance']}")
    print(f"Average Context Precision: {summary['average_context_precision']}")
    print("--------------------------")

    return {
        "summary_scores": summary,
        "detailed_results": detailed_results
    }
