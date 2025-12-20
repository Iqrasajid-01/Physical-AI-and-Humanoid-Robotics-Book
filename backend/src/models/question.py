from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


class QueryMode(str, Enum):
    full_book = "full_book"
    selected_text = "selected_text"


class QuestionBase(BaseModel):
    query_text: str = Field(..., min_length=5, max_length=2000)
    query_mode: QueryMode
    selected_text: Optional[str] = Field(None, min_length=10, max_length=5000)


class QuestionCreate(QuestionBase):
    book_id: Optional[UUID] = None  # Required for full_book mode
    session_id: Optional[UUID] = None


class Question(QuestionBase):
    id: UUID
    session_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    created_at: datetime
    processed_chunks_count: int = 0

    class Config:
        from_attributes = True


class QuestionResponse(BaseModel):
    id: UUID
    query_text: str
    query_mode: QueryMode
    created_at: datetime

    class Config:
        from_attributes = True