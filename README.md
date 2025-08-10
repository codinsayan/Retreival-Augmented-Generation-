# **Intelligent Document Query System**

## **Project Overview**

This project is a high-performance solution for the **LLM-Powered Intelligent Query–Retrieval System** challenge. The goal was to build a system capable of processing large, unstructured documents (like insurance policies) and answering complex, natural-language questions with high accuracy.

Our solution implements a sophisticated, multi-stage **Retrieval-Augmented Generation (RAG)** pipeline. This architecture goes far beyond simple keyword matching, leveraging multiple layers of AI-driven refinement to find the most precise information and generate factually grounded, relevant answers. Through rigorous testing with our evaluation framework, this system has demonstrated exceptional performance, achieving perfect scores across all key metrics.


## **Local Setup and Execution Guide**

### **Prerequisites**

* Python 3.10+  
* An active Python virtual environment (recommended)  
* API keys for **Pinecone** and **Google Gemini**

### **Step 1: Clone the Repository**

Clone the project files to your local machine.

### **Step 2: Install Dependencies**

Navigate to the project's root directory in your terminal and install all the required packages using the requirements.txt file.
```
pip install -r requirements.txt
```
### **Step 3: Configure Environment Variables**

1. In the project's root directory, create a file named `.env`.  
2. Add your API keys to this file. It should look like this:
 ```
   # Pinecone Configuration  
   PINECONE_API_KEY="YOUR_PINECONE_API_KEY_HERE"  
   PINECONE_INDEX_NAME="rag-challenge-index"  
   PINECONE_ENVIRONMENT="aws-us-east-1"

   # Google Gemini API Key  
   GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```
### **Step 4: Run the FastAPI Server**

Start the web server using uvicorn. Make sure to be in your project's `root folder`.
```
python -m uvicorn main:app --reload
```
The server will start and be accessible at `http://127.0.0.1:8000`.

### **Step 5: Test the API**

Open a **new** PowerShell terminal and use the following Invoke-RestMethod command to send a test request to your running server. This command uses the sample request from the problem statement.

```
POST http://127.0.0.1:8000/hackrx/run
Content-Type: application/json
Accept: application/json
Authorization: Bearer 55ec54fdbc2e11adcbec6345bfd72be68b31fccbc07f472ac530575a2e4b0a29

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}
```
The terminal will display the final JSON response from the server, containing the list of answers.

## **Our Approach: A Multi-Layered RAG Pipeline**

Our architecture is designed for maximum precision and reliability. It operates in three core phases for every user query:

### **1\. Pre-Retrieval: Query Understanding**

User questions are often brief and lack context. To overcome this, the first step is **Query Rewriting**. The user's question is sent to the Gemini LLM, which expands it into a more descriptive and detailed query. This enriches the prompt with relevant keywords and concepts, dramatically improving the effectiveness of the subsequent search steps.

<img width="1920" height="1080" alt="HackRx" src="https://github.com/user-attachments/assets/7921f811-14cf-4ebb-8823-e97be6da78c9" />

### **2\. Advanced Retrieval: Finding the Perfect Context**

This phase is designed to find the most relevant information with high recall and precision.

* **Hybrid Search:** We perform two types of searches in parallel:  
  * **Semantic Search (Pinecone):** A dense vector search to find document chunks that are *conceptually similar* to the query.  
  * **Keyword Search (In-Memory BM25):** A sparse vector search to find chunks containing *exact keywords* and terms.  
* **Reciprocal Rank Fusion (RRF):** The results from both search methods are intelligently merged into a single, unified list that leverages the strengths of both approaches.  
* **Post-Retrieval Re-ranking:** This fused list is then passed to a specialized **Cross-Encoder model**. This powerful model acts as a final, high-precision filter, analyzing the user's original question against each retrieved chunk to calculate a precise relevance score, ensuring the absolute best context is pushed to the top.

<img width="1920" height="1080" alt="HackRx (1)" src="https://github.com/user-attachments/assets/8c9af3a2-2f5c-42d5-be47-3869c360ab70" />


### **3\. Generation: Answering the Question**

The final, highly-curated context is passed to the **Gemini 1.5 Flash** model. The prompt is carefully engineered with a critical instruction: the model must generate its answer based **only** on the provided context. This ensures the final response is factually grounded in the source document and eliminates hallucinations.
