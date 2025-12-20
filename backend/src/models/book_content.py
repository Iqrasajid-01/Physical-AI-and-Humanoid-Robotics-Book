from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
import re


class BookContentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    author: str = Field(..., min_length=1, max_length=100)
    isbn: Optional[str] = Field(None, pattern=r"^(?:\d{10}|\d{13})?$")
    total_pages: int = Field(..., gt=0, le=1500)
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    metadata: Optional[Dict[str, Any]] = None


class BookContentCreate(BookContentBase):
    content: str = Field(..., min_length=100)  # At least 100 characters for a meaningful book


class BookContent(BookContentBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = 0

    class Config:
        from_attributes = True


class BookContentResponse(BaseModel):
    id: UUID
    title: str
    author: str
    isbn: Optional[str]
    total_pages: int
    language: str
    created_at: datetime
    updated_at: datetime
    chunk_count: int

    class Config:
        from_attributes = True