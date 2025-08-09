import os
import requests
import uuid
import tempfile
from contextlib import contextmanager
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from .document_parser import parse_document_to_sections
import re

# --- CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-challenge-index")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "aws-us-east-1") 

# --- INITIALIZATION ---
pc = Pinecone(api_key=PINECONE_API_KEY)

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIMENSION = embedding_model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dimension: {EMBEDDING_DIMENSION}")

# --- HELPER FUNCTIONS ---
@contextmanager
def temporary_pdf_file(url):
    temp_dir = tempfile.gettempdir()
    local_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.pdf")
    print(f"Downloading PDF from {url} to {local_filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
        yield local_filename
    finally:
        if os.path.exists(local_filename):
            os.remove(local_filename)
            print(f"Removed temporary file: {local_filename}")

# --- PINEONE INDEX MANAGEMENT ---
def get_pinecone_index():
    """Gets or creates a Pinecone serverless index."""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new serverless one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created successfully.")
    else:
        print(f"Found existing index: '{PINECONE_INDEX_NAME}'.")
    return pc.Index(PINECONE_INDEX_NAME)

# --- DENSE VECTOR UPSERT LOGIC ---
def upsert_dense_embeddings(index, sections, batch_size=50):
    """
    Creates dense vectors and upserts them to Pinecone.
    """
    print(f"Starting dense embedding generation and upsert for {len(sections)} sections...")
    
    for i in tqdm(range(0, len(sections), batch_size), desc="Upserting to Pinecone"):
        batch_sections = sections[i:i + batch_size]
        
        dense_texts = [section['full_content'] for section in batch_sections]
        dense_embeddings = embedding_model.encode(dense_texts, show_progress_bar=False).tolist()
        
        vectors_to_upsert = []
        for idx, section in enumerate(batch_sections):
            vector_id = f"dense_{i+idx}" # Use a simple unique ID
            metadata = {
                "document_name": section["document_name"],
                "page_number": int(section["page_number"]),
                "section_title": section["section_title"],
                "full_content": section["full_content"]
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": dense_embeddings[idx],
                "metadata": metadata
            })
            
        index.upsert(vectors=vectors_to_upsert)
        
    print("All sections have been embedded and upserted to Pinecone.")

# --- MAIN ORCHESTRATION ---
def process_and_embed_document(pdf_url: str):
    print("-" * 80)
    print(f"Starting full processing pipeline for document at: {pdf_url}")
    model_path = "models/heading_classifier_model.joblib"
    encoder_path = "models/label_encoder.joblib"

    with temporary_pdf_file(pdf_url) as local_pdf_path:
        sections = parse_document_to_sections(local_pdf_path, model_path, encoder_path)
        if not sections:
            print("No sections were parsed from the document. Aborting.")
            return

        index = get_pinecone_index()
        
        upsert_dense_embeddings(index, sections)

    print("Document processing pipeline finished successfully.")
    print("-" * 80)
