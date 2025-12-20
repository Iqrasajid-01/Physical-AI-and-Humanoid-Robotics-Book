from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class QuerySessionBase(BaseModel):
    user_id: Optional[UUID] = None
    book_content_id: UUID


class QuerySessionCreate(QuerySessionBase):
    pass


class QuerySession(QuerySessionBase):
    id: UUID
    created_at: datetime
    last_activity_at: datetime
    query_count: int = 0
    is_active: bool = True

    class Config:
        from_attributes = True


class QuerySessionResponse(BaseModel):
    id: UUID
    user_id: Optional[UUID]
    book_content_id: UUID
    created_at: datetime
    last_activity_at: datetime
    query_count: int
    is_active: bool

    class Config:
        from_attributes = True