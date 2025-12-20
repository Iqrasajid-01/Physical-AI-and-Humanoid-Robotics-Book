# Quickstart Guide: RAG Chatbot for Digital Books

**Feature**: 001-rag-chatbot | **Date**: 2025-12-17

## Overview

This guide provides step-by-step instructions to set up, configure, and run the RAG Chatbot for Digital Books locally. Follow these steps to get the application running on your development machine.

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for containerized setup)
- Cohere API key (free tier available)
- Qdrant Cloud account (free tier available)
- Neon Postgres account (free tier available)

## Setup Steps

### 1. Clone and Navigate to Project

```bash
# If you haven't already, navigate to your project directory
cd backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Copy the example environment file and update with your actual credentials:

```bash
cp .env.example .env
```

Edit `.env` file and add your API keys:

```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
DATABASE_URL=your_neon_postgres_connection_string_here
SECRET_KEY=your_secret_key_here
DEBUG=True
```

### 5. Initialize Database

```bash
# The application will automatically initialize the database on first run
# Or run initialization script if provided
```

### 6. Run the Application

#### Option A: Direct Python Execution

```bash
cd src
python app.py
```

#### Option B: Using Uvicorn

```bash
cd src
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Option C: Using Docker

```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## API Endpoints

Once running, the following endpoints will be available:

### Ingestion Endpoints
- `POST /api/v1/books/ingest` - Ingest a book for RAG
- `GET /api/v1/books/` - List all ingested books

### Query Endpoints
- `POST /api/v1/query/full-book` - Query the full book content
- `POST /api/v1/query/selected-text` - Query selected text only
- `POST /api/v1/chat` - General chat endpoint supporting both modes

### Health Check
- `GET /health` - Check API health status

## Basic Usage Examples

### 1. Ingest a Book

```bash
curl -X POST http://localhost:8000/api/v1/books/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Sample Book",
    "author": "Author Name",
    "content": "Full text content of the book goes here...",
    "language": "en"
  }'
```

### 2. Query Full Book Content

```bash
curl -X POST http://localhost:8000/api/v1/query/full-book \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "book-uuid-here",
    "query": "What is the main theme of this book?",
    "top_k": 7
  }'
```

### 3. Query Selected Text

```bash
curl -X POST http://localhost:8000/api/v1/query/selected-text \
  -H "Content-Type: application/json" \
  -d '{
    "selected_text": "Specific text that the user has highlighted...",
    "query": "Explain this concept in more detail?",
    "temp_collection_ttl_seconds": 300
  }'
```

## Frontend Widget Integration

To embed the chat widget in a digital book:

1. Include the widget CSS and JS files:
```html
<link rel="stylesheet" href="/static/widget/chat-widget.css">
<script src="/static/widget/chat-widget.js"></script>
```

2. Add the widget container to your HTML:
```html
<div id="rag-chatbot-widget"></div>
```

3. Initialize the widget:
```javascript
const chatWidget = new RagChatWidget({
  apiUrl: 'http://localhost:8000',
  bookId: 'book-uuid-here'
});
chatWidget.init();
```

## Configuration Options

### Environment Variables

- `DEBUG`: Enable/disable debug mode (default: False)
- `COHERE_API_KEY`: Your Cohere API key
- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant
- `DATABASE_URL`: Connection string for Neon Postgres
- `MAX_BOOK_PAGES`: Maximum pages allowed per book (default: 1500)
- `CHUNK_SIZE_TOKENS`: Average tokens per chunk (default: 400)
- `CHUNK_OVERLAP_TOKENS`: Overlap tokens between chunks (default: 100)
- `TOP_K_RETRIEVAL`: Number of chunks to retrieve (default: 7)
- `RESPONSE_TIMEOUT_SECONDS`: Timeout for API responses (default: 60)

### Runtime Parameters

The application accepts the following runtime parameters:

- `--host`: Host address to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 8000)
- `--workers`: Number of worker processes (default: 1 for development)

## Testing

### Run Unit Tests

```bash
pytest tests/unit/
```

### Run Integration Tests

```bash
pytest tests/integration/
```

### Run Full Test Suite

```bash
pytest tests/ --cov=src/ --cov-report=html
```

### Performance Testing

```bash
# Run performance tests to ensure <4s response times
python -m tests.performance.test_latency
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all API keys are correctly set in the `.env` file
   - Verify API key permissions and quotas

2. **Database Connection Issues**
   - Check that Neon Postgres connection string is correct
   - Ensure the database is accessible from your network

3. **Vector Database Issues**
   - Verify Qdrant URL and API key
   - Check that Qdrant collections are being created properly

4. **Slow Response Times**
   - Monitor token usage against Cohere free tier limits
   - Verify chunk size and retrieval parameters

### Useful Commands

```bash
# Check application logs
docker-compose logs -f

# View active collections in Qdrant
# Use Qdrant dashboard or API to inspect collections

# Check database tables
# Connect to Neon Postgres to verify table creation
```

## Next Steps

1. **Customize Configuration**: Adjust chunk size, retrieval parameters, and other settings for your specific use case
2. **Add Sample Data**: Ingest a test book to verify full functionality
3. **Integrate with Frontend**: Embed the widget in your digital book platform
4. **Monitor Performance**: Track response times and accuracy metrics
5. **Scale Deployment**: Move to production environment with load balancing

## Development Tips

- Use `DEBUG=True` during development for hot reloading
- Monitor API usage to stay within free tier limits
- Test with books of varying lengths to ensure performance consistency
- Verify citation accuracy by comparing responses to source material