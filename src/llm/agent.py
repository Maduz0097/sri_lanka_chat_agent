from llama_index.core.tools import FunctionTool
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from .tools import wikipedia_search, rag_search
from ..utils.storage import load_chat_history
from ..utils.database import AsyncSession, get_db
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Define tools
wiki_tool = FunctionTool.from_defaults(fn=wikipedia_search)
rag_tool = FunctionTool.from_defaults(fn=rag_search)
tools = [wiki_tool, rag_tool]

class CustomReActAgent(ReActAgent):
    def __init__(self, *args, chat_history: List[ChatMessage] = None, **kwargs):
        # Initialize memory buffer with provided chat history
        memory = ChatMemoryBuffer.from_defaults(
            chat_history=chat_history or [],
            llm=kwargs.get('llm')
        )
        # Initialize parent class with memory and other arguments
        super().__init__(*args, memory=memory, **kwargs)
        # Store chat history locally for compatibility
        self._chat_history: List[ChatMessage] = chat_history or []

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get()

    @chat_history.setter
    def chat_history(self, history: List[ChatMessage]) -> None:
        if not all(isinstance(msg, ChatMessage) for msg in history):
            raise ValueError("All history items must be ChatMessage objects")
        self.memory.set(history)
        self._chat_history = history

async def init_agent(db: AsyncSession) -> CustomReActAgent:
    """Initialize the agent with chat history from the database."""
    chat_history = await load_chat_history(db)
    return CustomReActAgent(
        tools=tools,
        llm=llm,
        chat_history=chat_history,
        verbose=True
    )