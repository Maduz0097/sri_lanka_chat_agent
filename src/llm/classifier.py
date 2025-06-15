from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import os
from dotenv import load_dotenv
import re
import asyncio
from ..utils.storage import save_chat_history
from typing import Tuple

# Load environment variables
load_dotenv()

# Initialize LLM for classification
llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

async def classify_greeting(query: str) -> bool:
    """Classify if the query contains a greeting."""
    prompt = f"Is the following query a greeting or contains a greeting? Respond with 'True' or 'False':\n{query}"
    response = await llm.acomplete(prompt)
    return response.text.strip().lower() == "true"

async def classify_sri_lanka(query: str) -> bool:
    """Classify if the query is related to Sri Lanka."""
    prompt = f"Is the following query related to Sri Lanka (e.g., its culture, history, geography)? Respond with 'True' or 'False':\n{query}"
    response = await llm.acomplete(prompt)
    return response.text.strip().lower() == "true"

async def handle_query(query: str, agent) -> Tuple[str, bool, bool]:
    """Handle the user query and return response, is_greeting, is_sri_lanka."""
    is_greeting = await classify_greeting(query)
    is_sri_lanka = await classify_sri_lanka(query)

    # Clean query by removing greetings
    cleaned_query = re.sub(r'^(hi|hello|hey|greetings|good morning|good evening)\s+', '', query, flags=re.IGNORECASE).strip()

    if not cleaned_query and is_greeting:
        return "Hello! I'm here to help with information about Sri Lanka. What's your question?", True, False

    if not is_sri_lanka:
        return "I'm specialized in answering questions about Sri Lanka. Please ask something related to Sri Lanka!", is_greeting, False

    # Query the agent
    response = await agent.achat(cleaned_query)
    return str(response), is_greeting, is_sri_lanka