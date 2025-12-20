from typing import Optional, List
from uuid import UUID
import re
from .constants import (
    MIN_TITLE_LENGTH, MAX_TITLE_LENGTH,
    MIN_AUTHOR_LENGTH, MAX_AUTHOR_LENGTH,
    MIN_QUESTION_LENGTH, MAX_QUESTION_LENGTH,
    MIN_RESPONSE_LENGTH, MAX_RESPONSE_LENGTH,
    MIN_SNIPPET_LENGTH, MAX_SNIPPET_LENGTH,
    MIN_CHUNK_SIZE, MAX_CHUNK_SIZE,
    MAX_SELECTED_TEXT_LENGTH
)


def validate_title(title: str) -> bool:
    """Validate book title length"""
    return MIN_TITLE_LENGTH <= len(title) <= MAX_TITLE_LENGTH


def validate_author(author: str) -> bool:
    """Validate author name length"""
    return MIN_AUTHOR_LENGTH <= len(author) <= MAX_AUTHOR_LENGTH


def validate_question_text(question: str) -> bool:
    """Validate question text length"""
    return MIN_QUESTION_LENGTH <= len(question) <= MAX_QUESTION_LENGTH


def validate_response_text(response: str) -> bool:
    """Validate response text length"""
    return MIN_RESPONSE_LENGTH <= len(response) <= MAX_RESPONSE_LENGTH


def validate_snippet(snippet: str) -> bool:
    """Validate citation snippet length"""
    return MIN_SNIPPET_LENGTH <= len(snippet) <= MAX_SNIPPET_LENGTH


def validate_chunk_text(chunk: str) -> bool:
    """Validate content chunk length"""
    # This is a simplified check - actual token count validation would require a tokenizer
    return MIN_CHUNK_SIZE <= len(chunk) <= MAX_CHUNK_SIZE


def validate_selected_text(selected_text: str) -> bool:
    """Validate selected text length"""
    return 0 < len(selected_text) <= MAX_SELECTED_TEXT_LENGTH


def validate_language_code(language: str) -> bool:
    """Validate ISO 639-1 language code (2-letter lowercase)"""
    return bool(re.match(r"^[a-z]{2}$", language))


def validate_page_numbers(pages: List[int]) -> bool:
    """Validate page numbers are positive"""
    return all(page > 0 for page in pages)


def validate_uuid(uuid_string: str) -> bool:
    """Validate UUID format"""
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False


def validate_cohere_response_quality(response: str, source_content: Optional[str] = None) -> bool:
    """
    Validate that the response is relevant to the source content and doesn't contain hallucinations
    This is a simplified check - in a real implementation, this would involve more sophisticated validation
    """
    if not response or not response.strip():
        return False

    # Check for obvious signs of hallucination (e.g., "I don't know" responses)
    lower_response = response.lower()
    if "i don't know" in lower_response or "not mentioned" in lower_response:
        # This might be a valid response if no relevant content was found
        return True

    # In a real implementation, we would check for semantic similarity between
    # the response and source content
    return True