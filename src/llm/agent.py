from llama_index.core.tools import FunctionTool
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .tools import wikipedia_search, rag_search
from ..utils.storage import load_chat_history
from ..utils.database import AsyncSession, get_db
from opentelemetry import trace
from typing import List
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


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
    """Initialize the agent with chat history from the database and local embeddings."""
    with tracer.start_as_current_span("init_agent"):
        try:
            # Set local embedding model
            with tracer.start_as_current_span("initialize_embeddings"):
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="D:\\agentic\\simple_agent\\models"
                )
                Settings.llm = llm
                logger.info("Initialized HuggingFaceEmbedding (all-MiniLM-L6-v2) for agent")

            # Verify embedding model
            if not isinstance(Settings.embed_model, HuggingFaceEmbedding):
                logger.error("Invalid embedding model detected: %s", type(Settings.embed_model))
                raise ValueError("Only HuggingFaceEmbedding is allowed")

            # Load chat history
            with tracer.start_as_current_span("load_chat_history"):
                chat_history = await load_chat_history(db)
                logger.info("Loaded %d chat history messages", len(chat_history))

            # Initialize agent
            agent = CustomReActAgent(
                tools=tools,
                llm=llm,
                chat_history=chat_history,
                verbose=True
            )
            logger.info("CustomReActAgent initialized successfully")
            return agent
        except Exception as e:
            logger.error("Failed to initialize agent: %s", str(e))
            tracer.get_current_span().record_exception(e)
            raise