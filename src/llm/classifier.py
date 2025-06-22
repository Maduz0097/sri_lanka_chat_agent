from llama_index.llms.groq import Groq
import os
from dotenv import load_dotenv
from ..utils.storage import load_chat_history
from llama_index.core.llms import ChatMessage
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession
import re
import logging
from typing import Tuple

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_TOKENS = 8192  # Groq llama3-70b-8192 limit
RESERVED_TOKENS = 1000  # Reserve for response and prompt overhead
MESSAGE_OVERHEAD = 10  # Tokens for formatting per message (e.g., "user: …\n")


# Initialize LLM for classification
llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Custom token counter
def count_tokens(text: str) -> int:
    """Estimate token count without a model tokenizer."""
    if not text:
        return 0
    # Split on whitespace to count words
    words = text.split()
    token_count = len(words)  # 1 token per word
    # Add tokens for punctuation and special characters
    punctuation = sum(1 for char in text if char in '.,!?;:"\'()[]{}')
    token_count += punctuation
    # Add 0.5 tokens for long words (>10 chars)
    long_words = sum(0.5 for word in words if len(word) > 10)
    token_count += int(long_words)
    return token_count

tracer = trace.get_tracer(__name__)


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

async def handle_query(query: str, agent, db: AsyncSession) -> Tuple[str, bool, bool]:
    """Handle the user query and return response, is_greeting, is_sri_lanka."""
    with tracer.start_as_current_span("handle_query", attributes={"query": query}):
        is_greeting = await classify_greeting(query)
        is_sri_lanka = await classify_sri_lanka(query)

        # Clean query by removing greetings
        cleaned_query = re.sub(r'^(hi|hello|hey|greetings|good morning|good evening)\s+', '', query, flags=re.IGNORECASE).strip()

        if not cleaned_query and is_greeting:
            return "Hello! I'm here to help with information about Sri Lanka. What's your question?", True, False

        if not is_sri_lanka:
            return "I'm specialized in answering questions about Sri Lanka. Please ask something related to Sri Lanka!", is_greeting, False

        # Load recent chat history for context
        with tracer.start_as_current_span("load_context"):
            history = await load_chat_history(db)
            context_messages = history[-5:]  # Last 5 messages for context
            context = "\n".join([f"{msg.role}: {msg.content}" for msg in context_messages])
            logger.info("Loaded %d messages for context", len(context_messages))

        # Combine context with query
        full_query = f"Conversation history:\n{context}\n\nUser: {cleaned_query}" if context else cleaned_query

        # Check token count
        with tracer.start_as_current_span("check_tokens"):
            tokens = count_tokens(full_query) + len(context_messages) * MESSAGE_OVERHEAD
            logger.info("Estimated token count for query: %d", tokens)
            if tokens > (MAX_TOKENS - RESERVED_TOKENS):
                # Truncate context by removing the oldest messages
                while tokens > (MAX_TOKENS - RESERVED_TOKENS) and context_messages:
                    context_messages.pop(0)
                    context = "\n".join([f"{msg.role}: {msg.content}" for msg in context_messages])
                    full_query = f"Conversation history:\n{context}\n\nUser: {cleaned_query}" if context else cleaned_query
                    tokens = count_tokens(full_query) + len(context_messages) * MESSAGE_OVERHEAD
                    logger.info("Truncated context, new token count: %d", tokens)

        # Query the agent
        with tracer.start_as_current_span("agent_query", attributes={"full_query": full_query}):
            response = await agent.achat(full_query)
            logger.info("Agent response generated, length: %d", len(str(response)))

        return str(response), is_greeting, is_sri_lanka