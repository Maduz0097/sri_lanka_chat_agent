from fastapi import FastAPI, HTTPException
from ..llm.classifier import handle_query
from ..utils.storage import save_chat_history, save_feedback
from .models import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse, HistoryResponse
from ..llm.agent import agent
from llama_index.core.llms import ChatMessage
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle a user query and return the chatbot's response."""
    global last_chat
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    response, is_greeting, is_sri_lanka = await handle_query(request.query)

    # Update chat history
    agent.chat_history.append(ChatMessage(role="user", content=request.query))
    agent.chat_history.append(ChatMessage(role="assistant", content=response))
    save_chat_history(agent.chat_history)

    # Store last chat for feedback
    last_chat = {"query": request.query, "response": response}

    return ChatResponse(
        response=response,
        is_greeting=is_greeting,
        is_sri_lanka=is_sri_lanka
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for the last chat response."""
    global last_chat
    if not last_chat:
        raise HTTPException(status_code=400, detail="No prior chat to provide feedback for")
    if request.rating.upper() not in ["Y", "N"]:
        raise HTTPException(status_code=400, detail="Rating must be 'Y' or 'N'")

    # Save feedback
    save_feedback(last_chat["query"], last_chat["response"], request.rating.upper())

    # Update chat history with feedback
    agent.chat_history.append(ChatMessage(role="user", content=f"Feedback: {request.rating.upper()}"))
    save_chat_history(agent.chat_history)

    return FeedbackResponse(message="Feedback submitted successfully")


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    """Retrieve the chat history."""
    history = [{"role": msg.role, "content": msg.content} for msg in agent.chat_history]
    return HistoryResponse(history=history)