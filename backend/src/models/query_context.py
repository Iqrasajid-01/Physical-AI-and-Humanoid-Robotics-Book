from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID


class QueryContext(BaseModel):
    """
    Represents the specific text context used for generating the response
    (either full book or selected text only)
    """
    book_id: Optional[UUID] = None
    selected_text: Optional[str] = None
    chunk_ids: List[UUID] = []
    context_type: str  # "full_book" or "selected_text"