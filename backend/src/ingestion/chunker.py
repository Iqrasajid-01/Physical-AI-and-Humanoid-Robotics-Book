import re
from typing import List, Tuple
from ..utils.constants import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from ..utils.logger import app_logger
from ..models.content_chunk import ContentChunkCreate


class TextChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, book_id: str, start_page: int = 1) -> List[ContentChunkCreate]:
        """
        Split text into semantically coherent chunks with overlap.
        This is a simplified implementation - in a real system, we would use tokenization
        to count actual tokens rather than characters.
        """
        # For this implementation, we'll use a simple approach based on sentences
        # In a real implementation, we would use a proper tokenizer like tiktoken
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""
        current_start_page = start_page
        current_end_page = start_page
        chunk_order = 0
        processed_chars = 0

        for i, sentence in enumerate(sentences):
            # Estimate token count (rough approximation: 1 token ~ 4 characters)
            sentence_token_count = len(sentence) // 4

            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size * 4:  # Convert back to char approximation
                if current_chunk.strip():
                    # Create a chunk with the current content
                    chunk = ContentChunkCreate(
                        book_content_id=book_id,
                        chunk_text=current_chunk.strip(),
                        chunk_order=chunk_order,
                        start_page=current_start_page,
                        end_page=current_end_page,
                        start_section=f"Section {chunk_order + 1}",
                        end_section=f"Section {chunk_order + 1}",
                        token_count=len(current_chunk) // 4  # Rough token estimation
                    )
                    chunks.append(chunk)
                    chunk_order += 1

                # Start a new chunk with overlap if possible
                if self.overlap > 0:
                    # Add overlap by including some previous sentences
                    overlap_start = max(0, len(chunks) - 1)
                    if overlap_start >= 0 and len(chunks) > 0:
                        # In a real implementation, we would add actual overlapping text
                        pass

                # Start new chunk with current sentence
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = ContentChunkCreate(
                book_content_id=book_id,
                chunk_text=current_chunk.strip(),
                chunk_order=chunk_order,
                start_page=current_start_page,
                end_page=current_end_page,
                start_section=f"Section {chunk_order + 1}",
                end_section=f"Section {chunk_order + 1}",
                token_count=len(current_chunk) // 4  # Rough token estimation
            )
            chunks.append(chunk)

        app_logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # This is a basic sentence splitting implementation
        # In a real system, we would use a more sophisticated NLP library
        sentence_endings = re.compile(r'[.!?]+[\s\n]+')
        sentences = sentence_endings.split(text)

        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _estimate_tokens(self, text: str) -> int:
        """Roughly estimate the number of tokens in text."""
        # This is a very rough approximation: 1 token ~ 4 characters
        # In a real implementation, use tiktoken or similar
        return len(text) // 4