from .database import ChatHistory, Feedback, get_db
from llama_index.core.llms import ChatMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List
import logging

logger = logging.getLogger(__name__)

async def load_chat_history(db: AsyncSession) -> List[ChatMessage]:
    """Load chat history from database."""
    try:
        result = await db.execute(
            text("SELECT role, content FROM chat_history ORDER BY timestamp ASC")
        )
        rows = result.fetchall()
        return [
            ChatMessage(role=row[0], content=row[1])
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
        return []

async def save_chat_history(db: AsyncSession, history: List[ChatMessage]) -> None:
    """Save chat history to database."""
    try:
        # Clear existing history to avoid duplicates
        await db.execute(text("DELETE FROM chat_history"))
        # Insert new history
        for msg in history:
            db.add(ChatHistory(role=msg.role, content=msg.content))
        await db.commit()
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        await db.rollback()
        raise

async def save_feedback(db: AsyncSession, query: str, response: str, rating: str) -> None:
    """Save feedback to database."""
    try:
        db.add(Feedback(query=query, response=response, rating=rating))
        await db.commit()
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        await db.rollback()
        raise