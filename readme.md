Sri Lanka Chatbot API
A Python-based chatbot API specializing in answering questions about Sri Lanka using Wikipedia as the primary source. Powered by LlamaIndex, Groq API (llama3-70b-8192), and FastAPI, it features dynamic query and greeting classification, local embeddings, persistent chat history, feedback collection, and a RESTful interface.
Features

Sri Lanka Focus: Answers only Sri Lanka-related queries (e.g., history, culture, cities) using Wikipedia data.
LLM-Based Interaction: Uses Groq's llama3-70b-8192 to classify queries and detect greetings (e.g., "Hi," "Good morning").
Greeting Handling: Responds to greetings and processes queries if combined (e.g., "Hello, what's the capital?").
Local Embeddings: Uses BAAI/bge-small-en for offline text vectorization.
Chat History: Persists to chat_history.json, accessible via /history.
Feedback Collection: Allows Y/N ratings via /feedback, stored in feedback_dataset.json.
API Endpoints: POST /chat, POST /feedback, GET /history.
Error Handling: Returns JSON error messages (400, 429, 503, 500).
OpenAPI Docs: Swagger UI at /docs and openapi.yaml.

Requirements

Python 3.8–3.11
Dependencies in requirements.txt
Groq API key (Groq console)

Setup

Clone or Download:
git clone <repository-url>
cd srilanka-chatbot

Or download srilanka_chatbot.py, requirements.txt, openapi.yaml, README.md.

Create Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

For GPU (PyTorch with CUDA 11.8):
pip install torch --index-url https://download.pytorch.org/whl/cu118


Set Up Environment:Create .env:
GROQ_API_KEY=your-grok-api-key-here


Run the API:
uvicorn srilanka_chatbot:app --host 0.0.0.0 --port 8000



Usage

Access the API:
Base URL: http://localhost:8000
Swagger UI: http://localhost:8000/docs


Endpoints:
POST /chat:curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"query": "Hello, what is the capital of Sri Lanka?"}'

Response:{
  "response": "Hey there! The capital of Sri Lanka is Colombo.",
  "is_greeting": true,
  "is_sri_lanka": true
}


POST /feedback:curl -X POST "http://localhost:8000/feedback" -H "Content-Type: application/json" -d '{"rating": "Y"}'

Response:{
  "message": "Feedback submitted successfully"
}


GET /history:curl -X GET "http://localhost:8000/history"

Response:{
  "history": [
    {"role": "user", "content": "Hello, what is the capital of Sri Lanka?"},
    {"role": "assistant", "content": "Hey there! The capital of Sri Lanka is Colombo."},
    {"role": "user", "content": "Feedback: Y"}
  ]
}




Example Flow (via Swagger UI or client):
Chat: {"query": "Hi"} → {"response": "Hello! I'm here to help...", "is_greeting": true, "is_sri_lanka": false}
Chat: {"query": "What's the capital?"} → {"response": "The capital of Sri Lanka is Colombo.", ...}
Feedback: {"rating": "Y"} → {"message": "Feedback submitted successfully"}



Troubleshooting

API Rate Limit (429):
Wait 30–60 seconds.
Verify Groq API key in .env.


Import Error:
Ensure llama-index-tools-wikipedia:pip install llama-index-tools-wikipedia


Fallback to WikipediaReader.


PyTorch Issues:
Verify:python -c "import torch; print(torch.__version__)"


Check CUDA (nvidia-smi) for GPU.


File I/O Errors:
Ensure write permissions for chat_history.json, feedback_dataset.json.


Dependencies:
Verify:pip list | grep -E "llama-index|sentence-transformers|torch|python-dotenv|aiohttp|fastapi|uvicorn"





Notes

Fine-Tuning: Feedback in feedback_dataset.json can be used for offline prompt engineering or fine-tuning with alternative models (Groq doesn’t support fine-tuning).
Persistence: Chat history and feedback are saved to JSON files. For production, consider a database.
Deployment: Use Gunicorn for production:gunicorn -w 4 -k uvicorn.workers.UvicornWorker srilanka_chatbot:app


OpenAPI: See openapi.yaml or /docs for endpoint details.

License
MIT License (or specify your preferred license).
