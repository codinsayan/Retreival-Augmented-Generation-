import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
from rank_bm25 import BM25Okapi
import json

# --- CONFIGURATION & INITIALIZATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-challenge-index")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2" 

genai.configure(api_key=GEMINI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
rerank_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
generation_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- IN-MEMORY BM25 SETUP ---
bm25_index = None
corpus = []

def build_bm25_index_from_file(filepath="golden_dataset.json"):
    """Builds the in-memory BM25 index from the document context."""
    global bm25_index, corpus
    print("Building in-memory BM25 index for keyword search...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    corpus = [item['ground_truth_context'] for item in data]
    tokenized_corpus = [simple_tokenizer(doc) for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("BM25 index built successfully.")

# --- HELPER FUNCTIONS ---
def simple_tokenizer(text):
    """A simple tokenizer to split text into words."""
    return re.findall(r'\b\w+\b', text.lower())

# --- PROMPT TEMPLATES ---
REWRITE_PROMPT_TEMPLATE = """
You are an expert at rewriting user questions to be more effective for a vector database search.
Your task is to take the user's question and expand it into a more descriptive, detailed query that contains more contextual keywords.
Do not answer the question. Only provide the rewritten query.

Original Question: {question}

Rewritten Query:
"""

# OPTIMIZED: Merged compression and answer generation into a single prompt
ANSWER_PROMPT_TEMPLATE = """
You are an expert assistant for a document query system. Your task is to try to answer the user's question based on the provided context.
Your answer must be concise and directly address the question, using only the most relevant sentences from the context.
Do not use any external knowledge. But use your common sense. If the context does not contain the answer, you must state that you don't have enough information.

CONTEXT:
---
{context}
---

QUESTION:
{question}

ANSWER:
"""

# --- QUERY PROCESSING, RETRIEVAL, AND GENERATION ---

def rewrite_query(question: str) -> str:
    print(f"Rewriting original query: '{question}'")
    prompt = REWRITE_PROMPT_TEMPLATE.format(question=question)
    try:
        response = generation_model.generate_content(prompt)
        rewritten = response.text.strip()
        print(f"Rewritten query: '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"Warning: Could not rewrite query due to API error. Using original query. Error: {e}")
        return question

def retrieve_and_rerank(question: str, top_k: int = 10, final_k: int = 3):
    """
    Performs hybrid search by combining in-memory BM25 and Pinecone semantic search,
    then re-ranks the fused results.
    """
    if bm25_index is None:
        build_bm25_index_from_file()

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    rewritten_question = rewrite_query(question)

    # 1. Keyword Search (BM25)
    print("Performing keyword search (BM25)...")
    tokenized_query = simple_tokenizer(rewritten_question)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_hits = [{'id': f'bm25_{i}', 'score': bm25_scores[i], 'metadata': {'full_content': corpus[i]}} for i in bm25_results]

    # 2. Semantic Search (Pinecone)
    print("Performing semantic search (Pinecone)...")
    query_embedding = embedding_model.encode(rewritten_question).tolist()
    pinecone_results = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    pinecone_hits = pinecone_results.get('matches', [])

    # 3. Reciprocal Rank Fusion (RRF)
    print("Fusing results with Reciprocal Rank Fusion...")
    fused_results = {}
    k = 60  # RRF ranking constant
    
    for rank, hit in enumerate(pinecone_hits):
        doc_id = hit['id']
        if doc_id not in fused_results:
            fused_results[doc_id] = {'score': 0, 'metadata': hit['metadata']}
        fused_results[doc_id]['score'] += 1 / (k + rank + 1)

    for rank, hit in enumerate(bm25_hits):
        doc_content = hit['metadata']['full_content']
        doc_id = f"content_{hash(doc_content)}"
        if doc_id not in fused_results:
            fused_results[doc_id] = {'score': 0, 'metadata': hit['metadata']}
        fused_results[doc_id]['score'] += 1 / (k + rank + 1)
    
    sorted_fused_results = sorted(fused_results.values(), key=lambda x: x['score'], reverse=True)
    
    # 4. Re-rank the top fused results
    # OPTIMIZED: Reduce the number of items to re-rank for speed
    rerank_pool_size = top_k + 5 
    top_fused_hits = [item['metadata']['full_content'] for item in sorted_fused_results[:rerank_pool_size]]
    if not top_fused_hits:
        return []
        
    print(f"Re-ranking the top {len(top_fused_hits)} fused results...")
    rerank_pairs = [[question, content] for content in top_fused_hits]
    scores = rerank_model.predict(rerank_pairs, show_progress_bar=False)
    
    reranked_results = sorted(zip(top_fused_hits, scores), key=lambda x: x[1], reverse=True)
    
    final_context = [content for content, score in reranked_results[:final_k]]
    print(f"Retrieved and re-ranked {len(final_context)} final context chunks.")
    return final_context

def generate_answer_with_gemini(question: str, context_chunks: list[str]) -> str:
    if not context_chunks:
        return "I could not find any relevant information in the document to answer this question."

    context_str = "\n\n".join(context_chunks)
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context_str, question=question)
    
    print("Sending prompt to Gemini for answer generation...")
    try:
        response = generation_model.generate_content(prompt)
        print("Successfully received response from Gemini.")
        return response.text.strip()
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return "Error: Could not generate an answer due to an API issue."

def answer_question(question: str) -> str:
    """
    Main orchestration function for the optimized RAG pipeline.
    """
    print(f"\n--- Answering question: '{question}' ---")
    
    # The context compression step is now implicitly handled by the final prompt
    retrieved_context = retrieve_and_rerank(question)
    answer = generate_answer_with_gemini(question, retrieved_context)
    
    print(f"Generated Answer: {answer}")
    return answer
