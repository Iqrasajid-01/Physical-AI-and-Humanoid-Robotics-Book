# Research: RAG Chatbot for Digital Books

**Feature**: 001-rag-chatbot | **Date**: 2025-12-17

## Overview

This research document addresses key technical decisions and validations required for implementing the RAG chatbot for digital books. It covers technology choices, architecture decisions, and best practices for the specified requirements.

## Key Decisions Resolved

### 1. Embedding Model Selection
- **Decision**: Cohere embed-english-v3.0
- **Rationale**: Chosen for high-quality English embeddings with 1024 dimensions, optimized for efficiency. While multilingual models exist, the focus is on English books, making this model optimal for the use case.
- **Alternatives considered**:
  - embed-multilingual-v3.0 (for multilingual support, but unnecessary overhead for English-focused application)
  - Older embed-english-light variants (lower quality embeddings)

### 2. Text Chunking Strategy
- **Decision**: Semantic chunking with 400-token average + 100-token overlap
- **Rationale**: Semantic chunking provides better relevance for RAG compared to fixed-size chunking, though it's more computationally intensive. The 400+100 approach balances context preservation with retrieval efficiency.
- **Alternatives considered**:
  - Fixed-size chunking (512 tokens) - simpler but potentially breaks semantic boundaries
  - Larger chunks (800+ tokens) - more context but poorer precision in retrieval

### 3. Retrieval Approach
- **Decision**: Vector similarity top-k=7 with optional Cohere rerank
- **Rationale**: Top-k=7 provides good balance between recall and performance. Cohere rerank will be used if free tier allows, with fallback to pure cosine similarity search.
- **Alternatives considered**:
  - Lower k values (3-5) - faster but potentially misses relevant results
  - Higher k values (10-15) - more comprehensive but slower

### 4. Selected-Text Isolation Method
- **Decision**: Temporary Qdrant collections per query
- **Rationale**: Maintains consistency with the main technology stack by using Qdrant for temporary collections rather than introducing FAISS as another dependency.
- **Alternatives considered**:
  - In-memory FAISS - lighter weight but adds dependency and reduces consistency
  - Qdrant point filtering - simpler but potentially less isolated

### 5. Frontend Framework Choice
- **Decision**: Vanilla JavaScript + Tailwind CSS
- **Rationale**: Minimal footprint for embeddable widget that won't conflict with host page JavaScript. Easy to embed without heavy dependencies.
- **Alternatives considered**:
  - React/Vue - powerful but overkill for widget, potential conflicts with host page
  - jQuery - older, heavier than needed

### 6. Error Handling Strategy
- **Decision**: Graceful fallbacks with retries and default responses
- **Rationale**: Ensures robustness against API rate limits and service failures while maintaining user experience.
- **Approach**:
  - Retry with exponential backoff on rate limit errors
  - Default responses when no relevant content found
  - Clear error messages for different failure modes

## Technology Validation Findings

### Cohere SDK Integration
- Confirmed embedding dimensions match Qdrant vector requirements (1024 dimensions for embed-english-v3.0)
- Verified API rate limits and free tier constraints are compatible with requirements
- Tested embedding generation and retrieval workflows with sample texts

### Qdrant Integration
- Confirmed vector storage and similarity search capabilities
- Validated metadata storage for book references (pages, chapters, sections)
- Tested collection management for temporary selected-text mode

### Performance Benchmarks
- Preliminary tests show <4s response times achievable with current architecture
- Chunking and embedding performance within acceptable ranges
- Retrieval and generation steps optimized for speed

## Architecture Sketch

### Ingestion Pipeline
```
Book Text → Chunking (400-token avg + 100 overlap) → Cohere Embedding → Qdrant Storage
                                                         ↓
                                                    Neon Metadata (pages/chapters)
```

### Query Pipeline
```
User Query → Cohere Embedding → Qdrant Retrieval (top-k=7) → Cohere Generation (RAG Context) → Response
```

### Selected-Text Mode
```
Selected Text → Temp Qdrant Collection → Cohere Embedding → Search in Temp Collection → Cohere Generation → Response
```

### System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Frontend       │ ←→ │  FastAPI Backend │ ←→ │  External APIs  │
│  Widget         │    │                  │    │                 │
│  (HTML/JS)      │    │ • app.py         │    │ • Cohere        │
└─────────────────┘    │ • Models         │    │ • Qdrant Cloud  │
                       │ • Ingestion      │    │ • Neon Postgres │
┌─────────────────┐    │ • Retrieval      │    └─────────────────┘
│  Storage        │ ←→ │ • Generation     │
│                 │    │ • DB Managers    │
│ • Qdrant (vec)  │    └──────────────────┘
│ • Neon (meta)   │
└─────────────────┘
```

## Testing Strategy

### Unit Tests
- Individual module testing (chunker output, embedder vectors, generator prompts)
- Pytest with assertion-based validation
- Mock external dependencies for isolation

### Integration Tests
- End-to-end ingestion + query workflows
- Mock book data for consistent testing
- Response relevance validation (manual + cosine similarity thresholds)

### Selected-Text Mode Tests
- Verify no access to main collection during selected-text queries
- Mock Qdrant calls to ensure isolation
- Confirm responses are grounded only in input text

### Performance Tests
- Query timing measurements
- <4s average response time validation
- Free-tier simulation testing

## Validation Checks

### Accuracy Validation
- 50+ test queries with 95%+ accuracy target
- Human evaluation rubric: relevance, factuality, no hallucination
- Manual scoring against ground truth

### Widget Integration
- HTML demo embedding validation
- Responsiveness across different screen sizes
- Compatibility with various book viewing platforms

### Deployment Validation
- Docker build/test cycle validation
- Environment configuration testing
- API endpoint accessibility verification

## Risk Mitigation

### API Rate Limits
- Implement retry mechanisms with exponential backoff
- Queue management for burst requests
- Graceful degradation when limits exceeded

### Large Book Handling
- Streaming processing for books up to 1,500 pages
- Memory management for large vector collections
- Progress tracking for long ingestion jobs

### Security Considerations
- API key protection via environment variables
- Rate limiting to prevent abuse
- Input validation to prevent injection attacks
- CORS configuration for secure embedding