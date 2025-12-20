from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import os
import time
from datetime import datetime
from uuid import UUID, uuid4
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import models and components
from .models.book_content import BookContentCreate, BookContentResponse
from .models.response import ResponseResponse
from .db.qdrant_manager import QdrantManager
from .db.neon_manager import NeonManager
from .ingestion.embedder import CohereEmbedder
from .ingestion.chunker import TextChunker
from .ingestion.processor import IngestionProcessor
from .retrieval.searcher import QdrantSearcher
from .retrieval.selector import SelectedTextSelector
from .generation.generator import CohereGenerator


class Settings(BaseSettings):
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", "")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    secret_key: str = os.getenv("SECRET_KEY", "default-secret-key-change-in-production")
    cluster_id: str = os.getenv("CLUSTER_ID", "")
    max_book_pages: int = int(os.getenv("MAX_BOOK_PAGES", "1500"))
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "400"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "7"))
    response_timeout_seconds: int = int(os.getenv("RESPONSE_TIMEOUT_SECONDS", "60"))

    class Config:
        env_file = ".env"


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class FullBookQueryRequest(BaseModel):
    book_id: UUID
    query: str = Field(..., min_length=5, max_length=2000)
    top_k: Optional[int] = Field(default=7, ge=1, le=20)
    enable_rerank: Optional[bool] = True


class SelectedTextQueryRequest(BaseModel):
    selected_text: str = Field(..., min_length=10, max_length=5000)
    query: str = Field(..., min_length=5, max_length=2000)
    temp_collection_ttl_seconds: Optional[int] = Field(default=300, ge=60, le=3600)


class UnifiedChatRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    book_id: Optional[UUID] = None  # Required for full-book mode
    selected_text: Optional[str] = Field(None, min_length=10, max_length=5000)  # Required for selected-text mode
    query_mode: str = Field(default="full_book", pattern=r"^(full_book|selected_text)$")  # "full_book" or "selected_text"
    session_id: Optional[UUID] = None
    top_k: Optional[int] = Field(default=7, ge=1, le=20)
    enable_rerank: Optional[bool] = True
    temp_collection_ttl_seconds: Optional[int] = Field(default=300, ge=60, le=3600)


class QueryResponse(BaseModel):
    response_text: str
    confidence_score: float
    citations: List[dict]
    processing_time_ms: int


app = FastAPI(
    title="RAG Chatbot API for Digital Books",
    description="API for a Retrieval-Augmented Generation chatbot that allows users to ask questions about digital book content",
    version="1.0.0"
)

# Initialize settings
settings = Settings()

# Initialize managers and services lazily (in a real app, these would be dependency injected)
def get_qdrant_manager():
    return QdrantManager(settings.qdrant_url, settings.qdrant_api_key)

def get_neon_manager():
    return NeonManager(settings.database_url)

def get_embedder():
    return CohereEmbedder(settings.cohere_api_key)

def get_chunker():
    return TextChunker()

def get_ingestion_processor():
    qdrant_mgr = get_qdrant_manager()
    neon_mgr = get_neon_manager()
    emb = get_embedder()
    chnk = get_chunker()
    return IngestionProcessor(qdrant_mgr, neon_mgr, emb, chnk)

def get_searcher():
    qdrant_mgr = get_qdrant_manager()
    emb = get_embedder()
    return QdrantSearcher(qdrant_mgr, emb)

def get_selector():
    qdrant_mgr = get_qdrant_manager()
    emb = get_embedder()
    chnk = get_chunker()
    return SelectedTextSelector(qdrant_mgr, emb, chnk)

def get_generator():
    return CohereGenerator(settings.cohere_api_key)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from .utils.logger import app_logger
    app_logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return {"detail": "Internal server error"}


# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    from .utils.logger import app_logger
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    app_logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")

    return response

# Add CORS middleware for embeddable widget support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is running"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/v1/books/ingest", response_model=BookContentResponse)
@limiter.limit("5/minute")
async def ingest_book(request: Request, book_data: BookContentCreate):
    """Ingest a book for RAG"""
    try:
        book_id = uuid4()
        ingestion_processor = get_ingestion_processor()
        success = await ingestion_processor.process_book(book_data, book_id)

        if success:
            # Return a mock response since we don't have full model implementation
            return BookContentResponse(
                id=book_id,
                title=book_data.title,
                author=book_data.author,
                isbn=book_data.isbn,
                total_pages=book_data.total_pages,
                language=book_data.language,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                chunk_count=0  # This would be calculated in a full implementation
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest book")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting book: {str(e)}")


@app.get("/api/v1/books/")
async def list_books():
    """List all ingested books"""
    # This would return a list of books from the database
    return {"books": []}


@app.post("/api/v1/query/full-book", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_full_book(request: Request, query_request: FullBookQueryRequest):
    """Query the full book content"""
    try:
        import time
        start_time = time.time()

        searcher = get_searcher()
        generator = get_generator()

        # Search in the book
        search_results = await searcher.search_in_book(
            book_id=query_request.book_id,
            query=query_request.query,
            top_k=query_request.top_k
        )

        if not search_results:
            # Generate fallback response if no results found
            response_data = await generator.generate_fallback_response(query_request.query)
            processing_time = int((time.time() - start_time) * 1000)
            response_data["processing_time_ms"] = processing_time
            return QueryResponse(**response_data)

        # Rerank if enabled
        if query_request.enable_rerank:
            search_results = await searcher.rerank_results(
                query=query_request.query,
                results=search_results,
                top_k=query_request.top_k
            )

        # Generate response using RAG
        response_data = await generator.generate_response(
            query=query_request.query,
            context_chunks=search_results
        )

        processing_time = int((time.time() - start_time) * 1000)
        response_data["processing_time_ms"] = processing_time

        return QueryResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying book: {str(e)}")


@app.post("/api/v1/query/selected-text", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_selected_text(request: Request, query_request: SelectedTextQueryRequest):
    """Query selected text only mode"""
    try:
        import time
        start_time = time.time()

        selector = get_selector()
        searcher = get_searcher()
        generator = get_generator()

        # Process the selected text by creating a temporary collection
        temp_collection_name = await selector.process_selected_text(
            query_request.selected_text,
            query_request.temp_collection_ttl_seconds
        )

        try:
            # Search in the temporary collection
            search_results = await searcher.search_in_selected_text(
                temp_collection_name=temp_collection_name,
                query=query_request.query,
                top_k=5  # Using fewer results for selected text mode
            )

            if not search_results:
                # Generate fallback response if no results found
                response_data = await generator.generate_fallback_response(query_request.query)
                processing_time = int((time.time() - start_time) * 1000)
                response_data["processing_time_ms"] = processing_time
                return QueryResponse(**response_data)

            # Generate response using RAG with the selected text context
            response_data = await generator.generate_response(
                query=query_request.query,
                context_chunks=search_results
            )

            processing_time = int((time.time() - start_time) * 1000)
            response_data["processing_time_ms"] = processing_time

            return QueryResponse(**response_data)
        finally:
            # Always clean up the temporary collection
            await selector.cleanup_temp_collection(temp_collection_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying selected text: {str(e)}")


@app.post("/api/v1/chat", response_model=QueryResponse)
@limiter.limit("30/minute")
async def unified_chat(request: Request, chat_request: UnifiedChatRequest):
    """Unified chat endpoint supporting both full-book and selected-text modes"""
    try:
        import time
        start_time = time.time()

        if chat_request.query_mode == "full_book":
            if not chat_request.book_id:
                raise HTTPException(status_code=400, detail="book_id is required for full-book mode")

            searcher = get_searcher()
            generator = get_generator()

            # Search in the specified book
            search_results = await searcher.search_in_book(
                book_id=chat_request.book_id,
                query=chat_request.query,
                top_k=chat_request.top_k
            )

            if not search_results:
                # Generate fallback response if no results found
                response_data = await generator.generate_fallback_response(chat_request.query)
                processing_time = int((time.time() - start_time) * 1000)
                response_data["processing_time_ms"] = processing_time
                return QueryResponse(**response_data)

            # Rerank if enabled
            if chat_request.enable_rerank:
                search_results = await searcher.rerank_results(
                    query=chat_request.query,
                    results=search_results,
                    top_k=chat_request.top_k
                )

            # Generate response using RAG
            response_data = await generator.generate_response(
                query=chat_request.query,
                context_chunks=search_results
            )

        elif chat_request.query_mode == "selected_text":
            if not chat_request.selected_text:
                raise HTTPException(status_code=400, detail="selected_text is required for selected-text mode")

            selector = get_selector()
            searcher = get_searcher()
            generator = get_generator()

            # Process the selected text by creating a temporary collection
            temp_collection_name = await selector.process_selected_text(
                chat_request.selected_text,
                chat_request.temp_collection_ttl_seconds
            )

            try:
                # Search in the temporary collection
                search_results = await searcher.search_in_selected_text(
                    temp_collection_name=temp_collection_name,
                    query=chat_request.query,
                    top_k=5  # Using fewer results for selected text mode
                )

                if not search_results:
                    # Generate fallback response if no results found
                    response_data = await generator.generate_fallback_response(chat_request.query)
                    processing_time = int((time.time() - start_time) * 1000)
                    response_data["processing_time_ms"] = processing_time
                    return QueryResponse(**response_data)

                # Generate response using RAG with the selected text context
                response_data = await generator.generate_response(
                    query=chat_request.query,
                    context_chunks=search_results
                )
            finally:
                # Always clean up the temporary collection
                await selector.cleanup_temp_collection(temp_collection_name)
        else:
            raise HTTPException(status_code=400, detail="Invalid query_mode. Use 'full_book' or 'selected_text'")

        processing_time = int((time.time() - start_time) * 1000)
        response_data["processing_time_ms"] = processing_time

        return QueryResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in unified chat: {str(e)}")


# Placeholder for other endpoints - will be implemented in later phases
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API for Digital Books", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)