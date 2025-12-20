from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID
from datetime import datetime


class Citation(BaseModel):
    page_numbers: List[int]
    chapter_titles: List[str]
    section_names: List[str]
    snippet: str = Field(..., min_length=10, max_length=500)
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ResponseBase(BaseModel):
    response_text: str = Field(..., min_length=10, max_length=5000)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    citations: List[Citation]
    retrieved_chunks_ids: List[UUID]
    processing_time_ms: int = Field(..., ge=0)
    has_fallback_response: bool = False


class ResponseCreate(ResponseBase):
    question_id: UUID


class Response(ResponseBase):
    id: UUID
    question_id: UUID
    generated_at: datetime

    class Config:
        from_attributes = True


class ResponseResponse(BaseModel):
    id: UUID
    response_text: str
    confidence_score: float
    citations: List[Citation]
    processing_time_ms: int
    generated_at: datetime

    class Config:
        from_attributes = True