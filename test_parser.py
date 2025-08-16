import requests
import json

# The URL where your FastAPI server is running (update if different, e.g., your ngrok URL)
API_URL = "https://9f7336005846.ngrok-free.app/hackrx/run"

# The list of PDF blob URLs to test with the new, verified links
TEST_PDFS = {
    "Image-based Document": "https://nlsblog.org/wp-content/uploads/2020/06/image-based-pdf-sample.pdf",
    "Complex Layout Document": "https://arxiv.org/pdf/2104.00228.pdf",
    "Blank Document": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "Standard Document": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
}

# Simple questions to ask for each document
QUESTIONS = [
    "What is the main subject of this document?",
    "Summarize the key findings or purpose in one sentence."
]

# The required authorization header
HEADERS = {
    "accept": "application/json",
    "Authorization": "Bearer 55ec54fdbc2e11adcbec6345bfd72be68b31fccbc07f472ac530575a2e4b0a29",
    "Content-Type": "application/json"
}

def run_test():
    """
    Sends a request to the RAG API for each test PDF and prints the response.
    """
    for name, url in TEST_PDFS.items():
        print(f"--- Testing: {name} ---")
        
        # Construct the JSON payload for the API request
        payload = {
            "documents": url,
            "questions": QUESTIONS
        }
        
        try:
            # Send the POST request with a generous timeout for the LLM
            response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=300) # 5-minute timeout
            
            # Check the response status
            response.raise_for_status()
            
            print(f"Status Code: {response.status_code} - OK")
            print("Response JSON:")
            # Pretty-print the JSON response
            print(json.dumps(response.json(), indent=2))
            
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            if e.response:
                print(f"Response Body: {e.response.text}")

        print("-" * (len(name) + 14) + "\n")

if __name__ == "__main__":
    run_test()
