import os
import json
from pathlib import Path
from dotenv import load_dotenv
import time

# --- SETUP ---
# Load environment variables first
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Import our RAG services AFTER loading env vars
from services.query_service import retrieve_and_rerank, generate_answer_with_gemini
from services.evaluation_service import run_evaluation_pipeline

# --- MAIN EXECUTION ---
def main():
    """
    Main function to orchestrate the evaluation process.
    """
    # 1. Load the golden dataset
    print("Loading golden dataset...")
    with open('golden_dataset.json', 'r', encoding='utf-8') as f:
        golden_dataset = json.load(f)
    print(f"Loaded {len(golden_dataset)} test items.")

    # 2. Run the RAG pipeline for each question in the dataset
    print("\n--- Running RAG Pipeline on Golden Dataset ---")
    results_with_context = []
    for item in golden_dataset:
        question = item['question']
        print(f"\nProcessing question: '{question}'")
        
        # a. Retrieve and re-rank context
        retrieved_context = retrieve_and_rerank(question)
        
        # b. Generate an answer
        generated_answer = generate_answer_with_gemini(question, retrieved_context)
        
        results_with_context.append({
            "question": question,
            "generated_answer": generated_answer,
            "retrieved_context": retrieved_context,
            "ground_truth_answer": item.get("ground_truth_answer"),
            "ground_truth_context": item.get("ground_truth_context")
        })
        # Small delay to avoid hitting API rate limits
        time.sleep(1) 

    # 3. Run the evaluation pipeline on the results
    evaluation_results = run_evaluation_pipeline(results_with_context)

    # 4. Save the detailed results to a file
    output_filename = "evaluation_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4)
        
    print(f"\nEvaluation complete. Detailed results saved to '{output_filename}'.")


if __name__ == "__main__":
    # Before running, ensure the document has been processed and embedded
    # For this script, we assume the Pinecone index is already populated.
    # You can run the main FastAPI app once to process the document first.
    main()
