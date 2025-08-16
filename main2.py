import os
import json
import requests
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uuid

# --- SETUP & INITIALIZATION ---
# Load environment variables from a .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set or empty.")
genai.configure(api_key=GEMINI_API_KEY)

# Use a model that supports file inputs
generation_model = genai.GenerativeModel('gemini-1.5-pro-latest')

app = FastAPI(
    title="Robust Single-Prompt RAG System",
    description="An API that intelligently processes any PDF and answers questions using a single, consolidated LLM call.",
    version="3.2.0" # Version updated for new logging logic
)

# --- Create a directory to store downloaded PDFs ---
PDF_DOWNLOAD_DIR = "downloaded_pdfs"
os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)


# --- CORE PROMPT TEMPLATE ---

MASTER_PROMPT_TEMPLATE_FILE = """
You are an expert AI assistant tasked with answering questions based *only* on the provided PDF document. The document may be a scan, contain images, or have complex layouts.

**INSTRUCTIONS:**
1.  Carefully analyze the entire PDF file provided.
2.  Analyze each question from the "QUESTIONS" list.
3.  For each question, find the precise answer within the document.
4.  Your final output MUST be a single, valid JSON object that adheres to the schema provided below.
5.  Each string in the "answers" list must correspond to the answer for the question in the same order.
6.  Do not use any external knowledge. If an answer cannot be found in the document (e.g., if it is blank), you must state that the information is not available in the provided text.

**JSON OUTPUT SCHEMA:**
```json
{{
  "answers": [
    "string"
  ]
}}
```

**QUESTIONS:**
---
{questions_list}
---

**IMPORTANT: Your entire response must be only the valid JSON object described in the schema, with no additional text, explanations, or markdown formatting.**

**FINAL JSON OUTPUT:**
"""


# --- API MODELS ---

class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- API ENDPOINT ---

@app.post("/hackrx/run", response_model=RunResponse, tags=["Query System"])
async def run_submission(
    request: RunRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Processes any document by sending the entire file to the Gemini API
    and answers all questions in a single call.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required.")

    # --- Print incoming request details ---
    print("\n--- New Request Received ---")
    print(f"Processing document URL: {request.documents}")
    print("Received questions:")
    for i, q in enumerate(request.questions):
        print(f"  {i+1}. {q}")
    print("--------------------------\n")
    
    # --- Download to a permanent folder and prepare log path ---
    request_id = str(uuid.uuid4())
    pdf_path = os.path.join(PDF_DOWNLOAD_DIR, f"{request_id}.pdf")
    log_path = os.path.join(PDF_DOWNLOAD_DIR, f"{request_id}_log.txt")
    
    try:
        # 1. Download the document to the new folder
        print(f"Downloading document to {pdf_path}...")
        with requests.get(str(request.documents), stream=True) as r:
            r.raise_for_status()
            with open(pdf_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")

        # --- ADDED: Write request details to a log file ---
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Source URL: {request.documents}\n")
            log_file.write("\n--- Questions ---\n")
            for i, q in enumerate(request.questions):
                log_file.write(f"{i+1}. {q}\n")
        print(f"Request details logged to {log_path}")
        # -----------------------------------------------
        
        # 2. Upload the file to the Gemini API
        print("Uploading PDF file to Gemini for multimodal analysis...")
        uploaded_file = genai.upload_file(path=pdf_path, display_name=os.path.basename(pdf_path))

        # 3. Format the questions and construct the prompt
        questions_list_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(request.questions)])
        final_prompt = MASTER_PROMPT_TEMPLATE_FILE.format(
            questions_list=questions_list_str
        )

        # 4. Send the prompt and the uploaded file to Gemini
        print("Sending consolidated file prompt to Gemini...")
        response = generation_model.generate_content([final_prompt, uploaded_file])

        # 5. Clean and parse the JSON response from the model
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text)
        
        print("Successfully received and parsed response from Gemini.")
        
        return RunResponse(**response_json)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

