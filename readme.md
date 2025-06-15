Sri Lanka Chatbot
A Python-based chatbot specializing in answering questions about Sri Lanka using Wikipedia as the primary source. Powered by LlamaIndex and the Groq API (llama3-70b-8192), it features dynamic query classification and greeting handling via the LLM, local embeddings with BAAI/bge-small-en, and a continuous chat interface.
Features

Sri Lanka Focus: Answers only Sri Lanka-related queries (e.g., history, culture, cities) using Wikipedia data.
LLM-Based Interaction: Uses Groq's llama3-70b-8192 to classify queries and detect greetings (e.g., "Hi," "Good morning").
Greeting Handling: Responds to greetings with friendly prompts and processes queries if combined (e.g., "Hello, what's the capital?").
Local Embeddings: Uses BAAI/bge-small-en for efficient, offline text vectorization.
Chat History: Maintains conversation context for follow-up questions.
Error Handling: Provides user-friendly messages for API errors, invalid inputs, and more.
Continuous Chat: Interactive loop with an "exit" command.

Requirements

Python 3.8–3.11
Dependencies listed in requirements.txt
Groq API key (free tier available at Groq console)

Setup

Clone or Download:
git clone <repository-url>
cd srilanka-chatbot

Or download srilanka_chatbot.py and requirements.txt.

Create Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

For GPU support (PyTorch with CUDA, e.g., 11.8):
pip install torch --index-url https://download.pytorch.org/whl/cu118


Set Up Environment:Create a .env file in the project root:
GROQ_API_KEY=your-grok-api-key-here

Obtain your key from Groq console.


Usage

Run the Chatbot:python srilanka_chatbot.py


Interact:
Ask Sri Lanka-related questions (e.g., "What is the capital of Sri Lanka?").
Use greetings (e.g., "Hi," "Good morning").
Type exit to quit.


Example Interaction:Welcome to the Sri Lanka Chatbot! Ask anything about Sri Lanka or just say hi (type 'exit' to quit).
You: Hi
Bot: Hello! I'm here to help with questions about Sri Lanka. What's on your mind?
You: What's the capital of Sri Lanka?
Bot: Hey there! The capital of Sri Lanka is Colombo.
You: exit
Goodbye!



Troubleshooting

API Rate Limit (429):
Wait 30–60 seconds and retry.
Check your Groq API key in .env.


Import Error:
Ensure llama-index-tools-wikipedia:pip install llama-index-tools-wikipedia


The chatbot falls back to WikipediaReader if unavailable.


PyTorch Issues:
Verify PyTorch:python -c "import torch; print(torch.__version__)"


Ensure CUDA compatibility (nvidia-smi) for GPU.


Dependencies:
Check installed packages:pip list | grep -E "llama-index|sentence-transformers|torch|python-dotenv|aiohttp"


Resolve conflicts with a new virtual environment.


