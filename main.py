import asyncio
import os
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
import aiohttp
from llama_index.core.llms import ChatMessage

# Try importing WikipediaToolSpec
try:
    from llama_index.tools.wikipedia import WikipediaToolSpec
    from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec
    WIKI_TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: WikipediaToolSpec not found. Falling back to WikipediaReader with VectorStoreIndex.")
    WIKI_TOOLS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure LlamaIndex settings
try:
    Settings.llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
except Exception as e:
    print(f"Bot: Error initializing settings: {str(e)}. Please check your setup.")
    exit(1)

# Initialize tools
tools = []
if WIKI_TOOLS_AVAILABLE:
    try:
        wiki_spec = WikipediaToolSpec()
        for tool in wiki_spec.to_tool_list():
            wrapped_tools = LoadAndSearchToolSpec.from_defaults(tool).to_tool_list()
            tools.extend(wrapped_tools)
    except Exception as e:
        print(f"Warning: Failed to initialize WikipediaToolSpec: {str(e)}. Falling back to WikipediaReader.")
        WIKI_TOOLS_AVAILABLE = False

if not WIKI_TOOLS_AVAILABLE:
    try:
        wiki_reader = WikipediaReader()
        documents = wiki_reader.load_data(pages=["Sri Lanka"])
        index = VectorStoreIndex.from_documents(documents)

        def rag_search(query: str) -> str:
            """Search indexed Sri Lanka Wikipedia page for relevant information"""
            try:
                query_engine = index.as_query_engine(similarity_top_k=2)
                response = query_engine.query(query)
                return str(response)
            except Exception as e:
                return f"Error retrieving information: {str(e)}"

        from llama_index.core.tools import FunctionTool
        tools.append(FunctionTool.from_defaults(fn=rag_search))
    except Exception as e:
        print(f"Bot: Error setting up WikipediaReader: {str(e)}. Please check your setup.")
        exit(1)

# Create ReActAgent with tools
try:
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=Settings.llm,
        verbose=True,
        chat_history=[]  # Initialize empty chat history
    )
except Exception as e:
    print(f"Bot: Error initializing ReActAgent: {str(e)}. Please check your setup.")
    exit(1)

# System prompt
system_prompt = (
    "You are a chatbot specializing in Sri Lanka. Answer questions only about Sri Lanka using Wikipedia as the source. "
    "Maintain chat history for context. If an error occurs, provide a user-friendly error message."
)
agent.system_prompt = system_prompt

async def classify_greeting(query: str) -> tuple[bool, str]:
    """Use LLM to classify if the query is a greeting"""
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
            return False, "Sorry, I've hit the Groq API rate limit while checking if your input is a greeting. Please try again."
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
    """Use LLM to classify if the query is about Sri Lanka"""
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
            return False, "Sorry, I've hit the Groq API rate limit while checking your query. Please try again."
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

async def handle_query(user_input: str) -> str:
    """Handle a single user query with error handling, including LLM-based greeting detection"""
    try:
        # Detect greetings using LLM
        is_greeting, greeting_error = await classify_greeting(user_input)
        if greeting_error:
            return greeting_error

        # Handle pure greetings (short inputs, likely only greetings)
        user_input_lower = user_input.lower().strip()
        if is_greeting and len(user_input_lower.split()) <= 3:  # Short inputs like "Hi" or "Good morning"
            response = "Hello! I'm here to help with questions about Sri Lanka. What's on your mind?"
            return response

        # Classify query for Sri Lanka relevance
        is_sri_lanka, error_message = await classify_query(user_input)
        if not is_sri_lanka:
            if is_greeting:
                return f"Hey there! {error_message}"
            return error_message

        # Process query
        response = await agent.achat(user_input)
        if is_greeting:
            response = f"Hey there! {str(response)}"
        return str(response)

    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            return "Sorry, I've hit the Groq API rate limit. Please try again in a moment."
        return f"Error connecting to Groq API: {str(e)}"
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        return f"Something went wrong while processing your query: {str(e)}. Please try again or rephrase your query."

async def chat_loop():
    """Run the continuous chatbot loop"""
    print("Welcome to the Sri Lanka Chatbot! Ask anything about Sri Lanka or just say hi (type 'exit' to quit).")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            if not user_input:
                print("Bot: Please enter a question or greeting.")
                continue

            # Handle the query
            response = await handle_query(user_input)
            print(f"Bot: {response}")

            # Update chat history
            agent.chat_history.append(ChatMessage(role="user", content=user_input))
            agent.chat_history.append(ChatMessage(role="assistant", content=response))

        except KeyboardInterrupt:
            print("\nBot: Interrupted. Type 'exit' to quit or continue with your question.")
        except Exception as e:
            print(f"Bot: Unexpected error: {str(e)}. Please try again.")

async def main():
    """Main function to start the chatbot"""
    try:
        await chat_loop()
    except Exception as e:
        print(f"Bot: Failed to start chatbot: {str(e)}. Please check your setup and try again.")

if __name__ == "__main__":
    asyncio.run(main())