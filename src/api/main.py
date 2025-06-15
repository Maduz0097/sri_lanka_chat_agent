from fastapi import FastAPI, HTTPException, Depends
from ..llm.classifier import handle_query
from ..utils.storage import save_chat_history, save_feedback, load_chat_history
from ..utils.database import get_db, init_db
from ..llm.agent import init_agent
from .models import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse, HistoryResponse
from llama_index.core.llms import ChatMessage
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Sri Lanka Chatbot API",
    description="A chatbot API specializing in Sri Lanka, powered by LlamaIndex and Groq LLM.",
    version="1.0.0"
)

# Global state for last chat response
last_chat: Optional[dict] = None


# Initialize database and agent on startup
@app.on_event("startup")
async def startup_event():
    await init_db()
    async for db in get_db():
        app.state.agent = await init_agent(db)
        break  # Use first session to initialize agent


# Dependency to get agent from app state
def get_agent():
    return app.state.agent


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db), agent=Depends(get_agent)):
    """Handle a user query and return the chatbot's response."""
    global last_chat
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    response, is_greeting, is_sri_lanka = await handle_query(request.query, agent=agent)

    # Update chat history
    agent.chat_history.append(ChatMessage(role="user", content=request.query))
    agent.chat_history.append(ChatMessage(role="assistant", content=response))
    await save_chat_history(db, agent.chat_history)

    # Store last chat for feedback
    last_chat = {"query": request.query, "response": response}

    return ChatResponse(
        response=response,
        is_greeting=is_greeting,
        is_sri_lanka=is_sri_lanka
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db), agent=Depends(get_agent)):
    """Submit feedback for the last chat response."""
    global last_chat
    if not last_chat:
        raise HTTPException(status_code=400, detail="No prior chat to provide feedback for")
    if request.rating.upper() not in ["Y", "N"]:
        raise HTTPException(status_code=400, detail="Rating must be 'Y' or 'N'")

    # Save feedback
    await save_feedback(db, last_chat["query"], last_chat["response"], request.rating.upper())

    # Update chat history with feedback
    agent.chat_history.append(ChatMessage(role="user", content=f"Feedback: {request.rating.upper()}"))
    await save_chat_history(db, agent.chat_history)

    return FeedbackResponse(message="Feedback submitted successfully")


@app.get("/history", response_model=HistoryResponse)
async def get_history(db: AsyncSession = Depends(get_db)):
    """Retrieve the chat history."""
    history = await load_chat_history(db)
    return HistoryResponse(history=[{"role": msg.role, "content": msg.content} for msg in history])