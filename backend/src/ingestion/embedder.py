import asyncio
import cohere
from typing import List, Dict, Any
from ..utils.logger import app_logger
from ..utils.constants import COHERE_EMBED_MODEL, COHERE_EMBED_DIMENSION


class CohereEmbedder:
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)
        self.model = COHERE_EMBED_MODEL
        self.dimension = COHERE_EMBED_DIMENSION

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.embed(
                    texts=[text],
                    model=self.model,
                    input_type="search_document"
                )
            )
            embedding = response.embeddings[0]

            if len(embedding) != self.dimension:
                app_logger.warning(f"Expected embedding dimension {self.dimension}, got {len(embedding)}")

            app_logger.info(f"Generated embedding for text of length {len(text)}")
            return embedding
        except Exception as e:
            app_logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            # Cohere has limits on batch size, so we'll process in chunks if needed
            all_embeddings = []

            # Process in batches of up to 96 texts (Cohere's typical limit)
            batch_size = 96
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda b=batch: self.client.embed(
                        texts=b,
                        model=self.model,
                        input_type="search_document"
                    )
                )

                batch_embeddings = response.embeddings
                all_embeddings.extend(batch_embeddings)

                app_logger.info(f"Generated embeddings for batch {i//batch_size + 1}, size: {len(batch)}")

            if all_embeddings and len(all_embeddings[0]) != self.dimension:
                app_logger.warning(f"Expected embedding dimension {self.dimension}, got {len(all_embeddings[0])}")

            app_logger.info(f"Generated embeddings for {len(texts)} texts")
            return all_embeddings
        except Exception as e:
            app_logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query (with different input type)"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.embed(
                    texts=[query],
                    model=self.model,
                    input_type="search_query"
                )
            )
            embedding = response.embeddings[0]

            if len(embedding) != self.dimension:
                app_logger.warning(f"Expected embedding dimension {self.dimension}, got {len(embedding)}")

            app_logger.info(f"Generated query embedding for: {query[:50]}...")
            return embedding
        except Exception as e:
            app_logger.error(f"Error generating query embedding: {str(e)}")
            raise