import os
from pathlib import Path
from dotenv import load_dotenv

# --- FIX: Robustly load environment variables ---
# Build the path to the .env file relative to this main.py file
# This ensures it works regardless of where you run the uvicorn command from
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from services.embedding_service import process_and_embed_document
from services.query_service import answer_question

app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System",
    description="An API that processes documents and answers questions using a RAG pipeline with Gemini.",
    version="1.0.0"
)

class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=RunResponse, tags=["Query System"])
async def run_submission(
    request: RunRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header with Bearer token is required.")

    try:
        process_and_embed_document(str(request.documents))

        answers = []
        for question in request.questions:
            generated_answer = answer_question(question)
            answers.append(generated_answer)
        
        return RunResponse(answers=answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "ok", "message": "Welcome to the RAG Query System API!"}
