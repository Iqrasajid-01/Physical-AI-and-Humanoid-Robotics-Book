from pydantic import BaseModel, Field
from typing import List
from uuid import UUID


class CitationBase(BaseModel):
    page_numbers: List[int] = Field(..., min_items=1)
    chapter_titles: List[str] = Field(default=[])
    section_names: List[str] = Field(default=[])
    snippet: str = Field(..., min_length=10, max_length=500)
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class CitationCreate(CitationBase):
    pass


class Citation(CitationBase):
    class Config:
        from_attributes = True


class CitationResponse(CitationBase):
    class Config:
        from_attributes = True