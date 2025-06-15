Sri Lanka Chatbot API
A Python-based chatbot API specializing in answering questions about Sri Lanka using Wikipedia as the primary source. Built with LlamaIndex, Groq API (llama3-70b-8192), and FastAPI, it features dynamic query and greeting classification, local embeddings, persistent chat history, feedback collection, and a modular design. The source code is available at https://github.com/Maduz0097/sri_lanka_chat_agent.git.
Features

Sri Lanka Focus: Answers only Sri Lanka-related queries using Wikipedia data.
LLM-Based Interaction: Uses Groq's llama3-70b-8192 for query and greeting classification.
Greeting Handling: Responds to greetings and processes combined queries.
Local Embeddings: Uses BAAI/bge-small-en for offline text vectorization.
Chat History: Persists to a PostgreSQL database, accessible via /history.
Feedback Collection: Y/N ratings via /feedback, stored in PostgreSQL.
API Endpoints: POST /chat, POST /feedback, GET /history.
Modular Design: Separates API, LLM, and utils logic.
OpenAPI Docs: Swagger UI at /docs and openapi.yaml.

Folder Structure
srilanka-chatbot/
├── src/
│   ├── api/          # FastAPI app, routes, models
│   ├── llm/          # LlamaIndex agent, tools, classification
│   ├── utils/        # Database and storage logic
│   └── __main__.py   # Optional entry point
├── requirements.txt  # Dependencies
├── openapi.yaml     # OpenAPI schema
├── README.md        # Documentation
├── .env.example     # Sample env file

Requirements

Python 3.8–3.11
PostgreSQL 13+
Dependencies in requirements.txt
Groq API key (Groq console)

Setup

Clone the Repository:
git clone https://github.com/Maduz0097/sri_lanka_chat_agent.git
cd srilanka-chatbot


Set Up PostgreSQL:

Install PostgreSQL (official guide).
Create a database:CREATE DATABASE srilanka_chatbot;


Note the database credentials (user, password, host, port).


Create Virtual Environment:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

For GPU (PyTorch with CUDA 11.8):
pip install torch --index-url https://download.pytorch.org/whl/cu118


Set Up Environment:Copy .env.example to .env:
cp .env.example .env

Edit .env:
GROQ_API_KEY=your-grok-api-key-here
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/srilanka_chatbot


Run the API:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000



Usage

Access the API:
Base URL: http://localhost:8000
Swagger UI: http://localhost:8000/docs


Endpoints:
POST /chat:curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"query": "Hello, what is the capital of Sri Lanka?"}'


POST /feedback:curl -X POST "http://localhost:8000/feedback" -H "Content-Type: application/json" -d '{"rating": "Y"}'


GET /history:curl -X GET "http://localhost:8000/history"




Example Flow:
Chat: {"query": "Hi"} → {"response": "Hello! I'm here to help...", "is_greeting": true, "is_sri_lanka": false}
Feedback: {"rating": "Y"} → {"message": "Feedback submitted successfully"}



Troubleshooting

Database Connection Errors:
Verify PostgreSQL is running: psql -U user -d srilanka_chatbot.
Check DATABASE_URL in .env.


ImportError:
Ensure you run from srilanka-chatbot/:uvicorn src.api.main:app


Reinstall dependencies: pip install -r requirements.txt.


API Rate Limit (429):
Wait 30–60 seconds.
Verify GROQ_API_KEY.


PyTorch Issues:
Verify: python -c "import torch; print(torch.__version__)".


Dependencies:
Check: pip list | grep -E "llama-index|sqlalchemy|asyncpg|fastapi|uvicorn".



Notes

Fine-Tuning: Feedback in the database supports offline prompt engineering.
Deployment: Use Gunicorn:gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app


Database Migrations: Use Alembic for schema changes (not implemented here but recommended).

License
MIT License.
