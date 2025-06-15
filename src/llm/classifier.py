from fastapi import HTTPException
from ..llm.agent import agent, Settings
from llama_index.core.llms import ChatMessage
import aiohttp

async def classify_greeting(query: str) -> tuple[bool, str]:
    """Use LLM to classify if the query is a greeting."""
    try:
        greeting_prompt = (
            f"Determine if the following input is a greeting (e.g., 'Hi,' 'Hello,' 'Good morning,' or similar expressions). "
            f"Respond with 'Yes' if it is a greeting (or contains a greeting), or 'No' if it is not. Input: '{query}'"
        )
        response = await Settings.llm.achat([ChatMessage(role="user", content=greeting_prompt)])
        is_greeting = response.message.content.strip().lower() == "yes"
        return is_greeting, ""
    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            return False, "Rate limit exceeded while checking if input is a greeting."
        return False, f"Error checking if input is a greeting: {str(e)}"
    except Exception as e:
        # Fallback to keyword check
        keywords = [
            "hello", "hi", "hey", "greetings", "how are you", "good morning",
            "good afternoon", "good evening", "howdy", "what's up"
        ]
        is_greeting = any(keyword in query.lower() for keyword in keywords)
        return is_greeting, ""

async def classify_query(query: str) -> tuple[bool, str]:
    """Use LLM to classify if the query is about Sri Lanka."""
    try:
        classification_prompt = (
            f"Determine if the following query is specifically about Sri Lanka (its geography, culture, history, cities, people, etc.). "
            f"Respond with 'Yes' if it is, or 'No' if it is not. Query: '{query}'"
        )
        response = await Settings.llm.achat([ChatMessage(role="user", content=classification_prompt)])
        is_sri_lanka = response.message.content.strip().lower() == "yes"
        if not is_sri_lanka:
            return False, (
                "I'm sorry, I can only answer questions about Sri Lanka. "
                "Please ask something related to Sri Lanka, like its capital, culture, or history."
            )
        return True, ""
    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            return False, "Rate limit exceeded while checking query."
        return False, f"Error checking query relevance: {str(e)}"
    except Exception as e:
        # Fallback to keyword check
        keywords = ["sri lanka", "srilanka", "colombo", "kandy", "galle", "sinhala", "tamil", "ceylon", "jaffna", "sinhalese"]
        is_sri_lanka = any(keyword in query.lower() for keyword in keywords)
        if not is_sri_lanka:
            return False, (
                "I'm sorry, I can only answer questions about Sri Lanka. "
                "Please ask something related to Sri Lanka, like its capital, culture, or history."
            )
        return True, ""

async def handle_query(user_input: str) -> tuple[str, bool, bool]:
    """Handle a single user query with error handling."""
    try:
        # Detect greetings
        is_greeting, greeting_error = await classify_greeting(user_input)
        if greeting_error:
            raise HTTPException(status_code=503, detail=greeting_error)

        # Handle pure greetings
        user_input_lower = user_input.lower().strip()
        if is_greeting and len(user_input_lower.split()) <= 3:
            return "Hello! I'm here to help with questions about Sri Lanka. What's on your mind?", is_greeting, False

        # Classify query for Sri Lanka relevance
        is_sri_lanka, error_message = await classify_query(user_input)
        if not is_sri_lanka:
            if is_greeting:
                return f"Hey there! {error_message}", is_greeting, False
            return error_message, is_greeting, False

        # Process query
        response = await agent.achat(user_input)
        if is_greeting:
            response = f"Hey there! {str(response)}"
        return str(response), is_greeting, is_sri_lanka

    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        raise HTTPException(status_code=503, detail=f"Error connecting to Groq API: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")