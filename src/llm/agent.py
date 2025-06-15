from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage
from ..utils.storage import load_chat_history
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables
load_dotenv()

# Custom ReActAgent with modifiable chat_history
class CustomReActAgent(ReActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chat_history: List[ChatMessage] = kwargs.get('chat_history', [])

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get the chat history."""
        return self._chat_history

    @chat_history.setter
    def chat_history(self, history: List[ChatMessage]) -> None:
        """Set the chat history."""
        if not all(isinstance(msg, ChatMessage) for msg in history):
            raise ValueError("All history items must be ChatMessage objects")
        self._chat_history = history

# Configure LlamaIndex settings
try:
    Settings.llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
except Exception as e:
    raise Exception(f"Error initializing settings: {str(e)}")

# Initialize agent with tools and chat history
try:
    from .tools import tools
    agent = CustomReActAgent.from_tools(
        tools=tools,
        llm=Settings.llm,
        verbose=True,
        chat_history=load_chat_history()
    )
    # System prompt
    agent.system_prompt = (
        "You are a chatbot specializing in Sri Lanka. Answer questions only about Sri Lanka using Wikipedia as the source. "
        "Maintain chat history for context. If an error occurs, provide a user-friendly error message."
    )
except Exception as e:
    raise Exception(f"Error initializing ReActAgent: {str(e)}")