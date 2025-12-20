# RAG Chatbot for Digital Books

A Retrieval-Augmented Generation (RAG) chatbot that allows digital book readers to ask questions about book content and receive accurate, context-aware answers with citations.

## Features

- Full-book Q&A: Ask questions about the entire book content
- Selected-text mode: Get answers based only on highlighted text
- Embeddable widget: Easy integration into digital book platforms
- Accurate citations: Responses include references to specific pages/chapters

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **Embeddings**: Cohere API
- **Vector Storage**: Qdrant
- **Metadata Storage**: Neon Postgres
- **Frontend**: Vanilla JavaScript + Tailwind CSS

## Requirements

- Python 3.11+
- Docker and Docker Compose (for containerized setup)
- Cohere API key
- Qdrant Cloud account
- Neon Postgres account

## Setup

1. Clone the repository
2. Navigate to the backend directory: `cd backend`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Copy the environment file: `cp .env.example .env`
7. Update `.env` with your API keys and configuration
8. Run the application: `uvicorn src.app:app --reload`

## Docker Setup

1. Navigate to the backend directory: `cd backend`
2. Build and run with Docker Compose: `docker-compose up --build`

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/books/ingest` - Ingest a book
- `GET /api/v1/books/` - List all books
- `POST /api/v1/query/full-book` - Query full book content
- `POST /api/v1/query/selected-text` - Query selected text only
- `POST /api/v1/chat` - Unified chat endpoint

## Environment Variables

- `COHERE_API_KEY`: Your Cohere API key
- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant
- `DATABASE_URL`: Connection string for Neon Postgres
- `SECRET_KEY`: Secret key for security
- `DEBUG`: Enable/disable debug mode (default: False)
- `MAX_BOOK_PAGES`: Maximum pages allowed per book (default: 1500)
- `CHUNK_SIZE_TOKENS`: Average tokens per chunk (default: 400)
- `CHUNK_OVERLAP_TOKENS`: Overlap tokens between chunks (default: 100)
- `TOP_K_RETRIEVAL`: Number of chunks to retrieve (default: 7)
- `RESPONSE_TIMEOUT_SECONDS`: Timeout for API responses (default: 60)

## Architecture

The application follows a modular architecture with distinct components for:
- Ingestion (text processing and embedding)
- Retrieval (vector search and document retrieval)
- Generation (response generation with RAG)
- Database management (Qdrant and Neon Postgres)

## Development

1. Run tests: `pytest`
2. Run tests with coverage: `pytest --cov=src/ --cov-report=html`
3. Format code: `black src/ tests/`
4. Check imports: `isort src/ tests/`
5. Lint code: `flake8 src/ tests/`
6. Type check: `mypy src/`

## Testing

The application includes several types of tests:

- **Unit tests**: Located in `tests/unit/`, testing individual components
- **Integration tests**: Located in `tests/integration/`, testing API endpoints
- **Contract tests**: Located in `tests/contract/`, testing API compliance
- **End-to-end tests**: Located in `tests/e2e/`, testing complete workflows

To run the full test suite:
```
pytest tests/
```

To run with coverage report:
```
pytest --cov=src/ --cov-report=html tests/
```

## License

[To be added]