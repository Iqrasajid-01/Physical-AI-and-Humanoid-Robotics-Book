import asyncio
from typing import List, Optional, Dict, Any
from uuid import UUID
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from ..utils.logger import app_logger
from ..utils.constants import (
    QDRANT_TIMEOUT, QDRANT_RETRIES, MAIN_COLLECTION_PREFIX,
    TEMP_COLLECTION_PREFIX, COHERE_EMBED_DIMENSION, TOP_K_RETRIEVAL
)


class QdrantManager:
    def __init__(self, url: str, api_key: Optional[str] = None):
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=QDRANT_TIMEOUT
        )
        self.retries = QDRANT_RETRIES

    async def create_collection(self, collection_name: str, recreate: bool = False):
        """Create a Qdrant collection for storing embeddings"""
        try:
            # Check if collection exists
            collections = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.get_collections().collections
            )

            collection_exists = any(col.name == collection_name for col in collections)

            if collection_exists and not recreate:
                app_logger.info(f"Collection {collection_name} already exists")
                return

            if collection_exists and recreate:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.delete_collection(collection_name)
                )

            # Create new collection
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=COHERE_EMBED_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
            )

            app_logger.info(f"Created collection {collection_name}")
        except Exception as e:
            app_logger.error(f"Error creating collection {collection_name}: {str(e)}")
            raise

    async def upsert_points(self, collection_name: str, points: List[PointStruct]):
        """Upsert points (vectors with metadata) into a collection"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
            )
            app_logger.info(f"Upserted {len(points)} points to {collection_name}")
        except Exception as e:
            app_logger.error(f"Error upserting points to {collection_name}: {str(e)}")
            raise

    async def search_points(self,
                           collection_name: str,
                           query_vector: List[float],
                           top_k: int = TOP_K_RETRIEVAL,
                           filters: Optional[models.Filter] = None):
        """Search for similar points in a collection"""
        try:
            search_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=filters
                )
            )
            app_logger.info(f"Found {len(search_result)} results in {collection_name}")
            return search_result
        except Exception as e:
            app_logger.error(f"Error searching in {collection_name}: {str(e)}")
            raise

    async def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.delete_collection(collection_name)
            )
            app_logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            app_logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            raise

    async def create_temp_collection(self, collection_suffix: str = None) -> str:
        """Create a temporary collection for selected-text mode"""
        import uuid
        suffix = collection_suffix or str(uuid.uuid4())[:8]  # Use random suffix if not provided
        collection_name = f"{TEMP_COLLECTION_PREFIX}{suffix}"
        await self.create_collection(collection_name)
        return collection_name

    async def delete_temp_collection(self, collection_name: str):
        """Delete a temporary collection"""
        if collection_name.startswith(TEMP_COLLECTION_PREFIX):
            await self.delete_collection(collection_name)

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            collection_info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.get_collection(collection_name)
            )
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "point_count": collection_info.points_count
            }
        except Exception as e:
            app_logger.error(f"Error getting collection info for {collection_name}: {str(e)}")
            raise

    async def close(self):
        """Close the Qdrant client connection"""
        try:
            # The QdrantClient doesn't have a close method, but we can log the event
            app_logger.info("Qdrant client connection closed")
        except Exception as e:
            app_logger.error(f"Error closing Qdrant client: {str(e)}")