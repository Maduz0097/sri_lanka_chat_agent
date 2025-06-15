import json
import os
from llama_index.core.llms import ChatMessage
from typing import List

HISTORY_FILE = "chat_history.json"
FEEDBACK_FILE = "feedback_dataset.json"

def load_chat_history() -> List[ChatMessage]:
    """Load chat history from file."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                return [ChatMessage(**msg) for msg in data]
        return []
    except Exception as e:
        print(f"Error loading chat history: {str(e)}")
        return []

def save_chat_history(history: List[ChatMessage]) -> None:
    """Save chat history to file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump([msg.dict() for msg in history], f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")

def save_feedback(query: str, response: str, rating: str) -> None:
    """Save user feedback to dataset."""
    try:
        data = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r') as f:
                data = json.load(f)
        data.append({"query": query, "response": response, "rating": rating})
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")