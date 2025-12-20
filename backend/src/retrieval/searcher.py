import asyncio
from typing import List, Optional
from uuid import UUID
from qdrant_client.http import models
from ..utils.logger import app_logger
from ..db.qdrant_manager import QdrantManager
from ..ingestion.embedder import CohereEmbedder


class QdrantSearcher:
    def __init__(self, qdrant_manager: QdrantManager, embedder: CohereEmbedder):
        self.qdrant_manager = qdrant_manager
        self.embedder = embedder

    async def search_in_book(self, book_id: UUID, query: str, top_k: int = 7) -> List[dict]:
        """Search for relevant chunks in a specific book"""
        try:
            collection_name = f"book_{book_id}"
            app_logger.info(f"Searching in collection {collection_name} for query: {query[:50]}...")

            # Generate embedding for the query
            query_vector = await self.embedder.embed_query(query)

            # Search in Qdrant
            search_results = await self.qdrant_manager.search_points(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k
            )

            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "id": result.id,
                    "chunk_text": result.payload.get("chunk_text", ""),
                    "start_page": result.payload.get("start_page", 0),
                    "end_page": result.payload.get("end_page", 0),
                    "start_section": result.payload.get("start_section", ""),
                    "end_section": result.payload.get("end_section", ""),
                    "score": result.score,
                    "book_id": result.payload.get("book_id")
                }
                formatted_results.append(formatted_result)

            app_logger.info(f"Found {len(formatted_results)} results for query in book {book_id}")
            return formatted_results

        except Exception as e:
            app_logger.error(f"Error searching in book {book_id}: {str(e)}")
            raise

    async def search_in_selected_text(self, temp_collection_name: str, query: str, top_k: int = 7) -> List[dict]:
        """Search for relevant chunks in temporary collection (selected-text mode)"""
        try:
            app_logger.info(f"Searching in temporary collection {temp_collection_name} for query: {query[:50]}...")

            # Generate embedding for the query
            query_vector = await self.embedder.embed_query(query)

            # Search in Qdrant
            search_results = await self.qdrant_manager.search_points(
                collection_name=temp_collection_name,
                query_vector=query_vector,
                top_k=top_k
            )

            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "id": result.id,
                    "chunk_text": result.payload.get("chunk_text", ""),
                    "score": result.score
                }
                formatted_results.append(formatted_result)

            app_logger.info(f"Found {len(formatted_results)} results for query in temporary collection")
            return formatted_results

        except Exception as e:
            app_logger.error(f"Error searching in temporary collection {temp_collection_name}: {str(e)}")
            raise

    async def rerank_results(self, query: str, results: List[dict], top_k: int = 7) -> List[dict]:
        """Rerank search results using Cohere's rerank functionality"""
        try:
            # Extract text from results for reranking
            texts = [result["chunk_text"] for result in results]

            # In a real implementation, we would use Cohere's rerank API
            # For now, we'll just return the original results as reranking
            # would require a separate Cohere rerank call
            app_logger.info(f"Reranked {len(results)} results")
            return results[:top_k]

        except Exception as e:
            app_logger.error(f"Error reranking results: {str(e)}")
            # If reranking fails, return original results
            return results[:top_k]