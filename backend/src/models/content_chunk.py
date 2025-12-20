from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime


class ContentChunkBase(BaseModel):
    book_content_id: UUID
    chunk_text: str = Field(..., min_length=50, max_length=10000)  # Length in characters, not tokens
    chunk_order: int
    start_page: int
    end_page: int
    start_section: str
    end_section: str
    token_count: int
    embedding_vector: Optional[list] = None  # Will be stored in Qdrant, not in Neon


class ContentChunkCreate(ContentChunkBase):
    pass


class ContentChunk(ContentChunkBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class ContentChunkResponse(BaseModel):
    id: UUID
    book_content_id: UUID
    chunk_text: str
    chunk_order: int
    start_page: int
    end_page: int
    start_section: str
    end_section: str
    token_count: int
    created_at: datetime

    class Config:
        from_attributes = True