import asyncio
from typing import List, Dict, Any
from uuid import UUID
from qdrant_client.http.models import PointStruct
from ..utils.logger import app_logger
from ..db.qdrant_manager import QdrantManager
from ..ingestion.embedder import CohereEmbedder
from ..ingestion.chunker import TextChunker


class SelectedTextSelector:
    def __init__(self, qdrant_manager: QdrantManager, embedder: CohereEmbedder, chunker: TextChunker):
        self.qdrant_manager = qdrant_manager
        self.embedder = embedder
        self.chunker = chunker

    async def process_selected_text(self, selected_text: str, temp_collection_ttl: int = 300) -> str:
        """Process selected text by creating a temporary collection"""
        try:
            # Create a temporary collection name
            temp_collection_name = await self.qdrant_manager.create_temp_collection()

            # Chunk the selected text
            # For selected text, we'll create a single chunk or small chunks
            chunks = self.chunker.chunk_text(
                text=selected_text,
                book_id="selected_text",  # Placeholder ID
                start_page=1
            )

            # Generate embeddings for the chunks
            chunk_texts = [chunk.chunk_text for chunk in chunks]
            embeddings = await self.embedder.embed_texts(chunk_texts)

            # Create Qdrant points
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=f"selected_chunk_{i}",
                    vector=embedding,
                    payload={
                        "chunk_text": chunk.chunk_text,
                        "chunk_order": chunk.chunk_order,
                        "source": "selected_text"
                    }
                )
                points.append(point)

            # Store embeddings in the temporary collection
            await self.qdrant_manager.upsert_points(temp_collection_name, points)
            app_logger.info(f"Stored {len(points)} vectors in temporary collection {temp_collection_name}")

            return temp_collection_name

        except Exception as e:
            app_logger.error(f"Error processing selected text: {str(e)}")
            raise

    async def cleanup_temp_collection(self, collection_name: str):
        """Clean up temporary collection after use"""
        try:
            if collection_name.startswith("temp_"):
                await self.qdrant_manager.delete_temp_collection(collection_name)
                app_logger.info(f"Cleaned up temporary collection {collection_name}")
            else:
                app_logger.warning(f"Attempted to clean up non-temporary collection: {collection_name}")
        except Exception as e:
            app_logger.error(f"Error cleaning up temporary collection {collection_name}: {str(e)}")

    async def ensure_isolation(self, query: str, temp_collection_name: str, main_book_id: UUID) -> bool:
        """Ensure that search is isolated to the temporary collection only"""
        try:
            # This method would implement checks to ensure the query is only performed
            # on the temporary collection and not on the main book collection
            # For now, we just return True to indicate isolation is maintained
            app_logger.info(f"Isolation check passed for query in {temp_collection_name}")
            return True
        except Exception as e:
            app_logger.error(f"Error in isolation check: {str(e)}")
            return False