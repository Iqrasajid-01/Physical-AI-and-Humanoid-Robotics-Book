import asyncio
from typing import List, Optional
from uuid import UUID
from ..utils.logger import app_logger
from ..utils.validators import validate_chunk_text
from ..models.book_content import BookContentCreate
from ..models.content_chunk import ContentChunkCreate
from ..db.qdrant_manager import QdrantManager
from ..db.neon_manager import NeonManager
from ..ingestion.chunker import TextChunker
from ..ingestion.embedder import CohereEmbedder
from qdrant_client.http.models import PointStruct


class IngestionProcessor:
    def __init__(self, qdrant_manager: QdrantManager, neon_manager: NeonManager,
                 embedder: CohereEmbedder, chunker: TextChunker):
        self.qdrant_manager = qdrant_manager
        self.neon_manager = neon_manager
        self.embedder = embedder
        self.chunker = chunker

    async def process_book(self, book_data: BookContentCreate, book_id: UUID) -> bool:
        """Process a book by chunking, embedding, and storing in databases"""
        try:
            app_logger.info(f"Starting ingestion process for book: {book_data.title}")

            # 1. Chunk the text
            chunks = self.chunker.chunk_text(
                text=book_data.content,
                book_id=str(book_id),
                start_page=1
            )
            app_logger.info(f"Text chunked into {len(chunks)} chunks")

            # 2. Generate embeddings for all chunks
            chunk_texts = [chunk.chunk_text for chunk in chunks]
            embeddings = await self.embedder.embed_texts(chunk_texts)
            app_logger.info(f"Generated embeddings for {len(embeddings)} chunks")

            # 3. Create Qdrant points
            collection_name = f"book_{book_id}"
            await self.qdrant_manager.create_collection(collection_name)

            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(chunk.id) if hasattr(chunk, 'id') else f"{book_id}_chunk_{i}",
                    vector=embedding,
                    payload={
                        "book_id": str(book_id),
                        "chunk_order": chunk.chunk_order,
                        "start_page": chunk.start_page,
                        "end_page": chunk.end_page,
                        "start_section": chunk.start_section,
                        "end_section": chunk.end_section,
                        "chunk_text": chunk.chunk_text,
                        "token_count": chunk.token_count
                    }
                )
                points.append(point)

            # 4. Store embeddings in Qdrant
            await self.qdrant_manager.upsert_points(collection_name, points)
            app_logger.info(f"Stored {len(points)} vectors in Qdrant collection {collection_name}")

            # 5. Store metadata in Neon (just basic info, chunks are in Qdrant)
            metadata = {
                "title": book_data.title,
                "author": book_data.author,
                "isbn": book_data.isbn,
                "total_pages": book_data.total_pages,
                "language": book_data.language,
                "chunk_count": len(chunks)
            }

            success = await self.neon_manager.save_book_metadata(
                book_id=book_id,
                title=book_data.title,
                author=book_data.author,
                total_pages=book_data.total_pages,
                language=book_data.language,
                metadata=book_data.metadata or {}
            )

            if success:
                app_logger.info(f"Book {book_data.title} successfully ingested")
                return True
            else:
                app_logger.error(f"Failed to save book metadata for {book_data.title}")
                return False

        except Exception as e:
            app_logger.error(f"Error processing book {book_data.title}: {str(e)}")
            raise

    async def validate_and_clean_text(self, text: str) -> str:
        """Validate and clean input text before processing"""
        # Remove excessive whitespace
        cleaned_text = ' '.join(text.split())

        # Additional cleaning can be added here
        # For example, removing special characters, normalizing encoding, etc.

        app_logger.info(f"Cleaned text from {len(text)} to {len(cleaned_text)} characters")
        return cleaned_text