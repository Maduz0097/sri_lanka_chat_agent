from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    is_greeting: bool
    is_sri_lanka: bool

class FeedbackRequest(BaseModel):
    rating: str  # Y/N

class FeedbackResponse(BaseModel):
    message: str

class HistoryResponse(BaseModel):
    history: List[dict]