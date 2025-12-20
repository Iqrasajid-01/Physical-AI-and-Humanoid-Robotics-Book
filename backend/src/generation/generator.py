import asyncio
import cohere
from typing import List, Dict, Any
from ..utils.logger import app_logger
from ..utils.constants import COHERE_GENERATE_MODEL
from ..models.citation import Citation


class CohereGenerator:
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)
        self.model = COHERE_GENERATE_MODEL

    async def generate_response(self, query: str, context_chunks: List[Dict[str, Any]],
                              max_tokens: int = 500) -> Dict[str, Any]:
        """Generate a response using RAG (Retrieval-Augmented Generation)"""
        try:
            # Format the context from retrieved chunks
            context_text = "\n\n".join([chunk["chunk_text"] for chunk in context_chunks])

            # Create a prompt that includes the context and query
            prompt = f"""
            Based on the following context, please answer the question.
            If the answer is not in the context, please say so clearly.

            Context:
            {context_text}

            Question: {query}

            Answer:
            """

            # Generate the response using Cohere
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.3,  # Lower temperature for more factual responses
                    stop_sequences=["\n\nQuestion:", "\n\nContext:"]
                )
            )

            generated_text = response.generations[0].text.strip()

            # Calculate confidence score based on response quality
            confidence_score = self._calculate_confidence_score(generated_text, context_chunks)

            # Generate citations from the context chunks
            citations = self._generate_citations(context_chunks)

            result = {
                "response_text": generated_text,
                "confidence_score": confidence_score,
                "citations": citations,
                "retrieved_chunks_ids": [chunk.get("id") for chunk in context_chunks if chunk.get("id")]
            }

            app_logger.info(f"Generated response for query: {query[:50]}...")
            return result

        except Exception as e:
            app_logger.error(f"Error generating response: {str(e)}")
            # Return a fallback response
            return {
                "response_text": "I'm sorry, I couldn't generate a response at this time. Please try again later.",
                "confidence_score": 0.0,
                "citations": [],
                "retrieved_chunks_ids": [],
                "has_fallback_response": True
            }

    def _calculate_confidence_score(self, response: str, context_chunks: List[Dict[str, Any]]) -> float:
        """Calculate a basic confidence score based on response characteristics"""
        # This is a simplified confidence calculation
        # In a real implementation, this would be more sophisticated

        if not response or response.startswith("I'm sorry") or "not mentioned" in response.lower():
            return 0.2  # Low confidence for "I don't know" type responses

        # Calculate how much of the response is supported by context
        response_lower = response.lower()
        context_text = " ".join([chunk["chunk_text"].lower() for chunk in context_chunks])

        # Count overlapping words between response and context
        response_words = set(response_lower.split())
        context_words = set(context_text.split())
        overlap = len(response_words.intersection(context_words))

        # Calculate a basic score based on overlap
        if len(response_words) == 0:
            return 0.0

        overlap_ratio = overlap / len(response_words)
        confidence = min(0.8 + overlap_ratio * 0.2, 1.0)  # Cap at 1.0

        return confidence

    def _generate_citations(self, context_chunks: List[Dict[str, Any]]) -> List[Citation]:
        """Generate citations from context chunks"""
        citations = []

        for chunk in context_chunks:
            citation = Citation(
                page_numbers=[chunk.get("start_page", 1)],  # Simplified - in reality, might span multiple pages
                chapter_titles=[chunk.get("start_section", "Unknown")],
                section_names=[chunk.get("end_section", "Unknown")],
                snippet=chunk["chunk_text"][:200] + "..." if len(chunk["chunk_text"]) > 200 else chunk["chunk_text"],  # First 200 chars as snippet
                relevance_score=chunk.get("score", 0.5)  # Use the search score as relevance score
            )
            citations.append(citation)

        return citations

    async def generate_fallback_response(self, query: str) -> Dict[str, Any]:
        """Generate a fallback response when no relevant context is found"""
        app_logger.info(f"Generating fallback response for query: {query}")

        return {
            "response_text": f"I couldn't find relevant information in the book to answer your question: '{query}'. The book might not contain information about this topic.",
            "confidence_score": 0.1,
            "citations": [],
            "retrieved_chunks_ids": [],
            "has_fallback_response": True
        }