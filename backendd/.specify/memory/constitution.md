<!--
Sync Impact Report:
- Version change: N/A (initial version) → 1.0.0
- List of modified principles: N/A (initial creation)
- Added sections: All principles and sections (initial creation)
- Removed sections: None
- Templates requiring updates: ✅ .specify/templates/plan-template.md - Constitution Check should align with new principles
- Templates requiring updates: ✅ .specify/templates/spec-template.md - Requirements section should align with new constraints
- Templates requiring updates: ✅ .specify/templates/tasks-template.md - Task categorization should reflect new principles
- Follow-up TODOs: None
-->

# Integrated Retrieval-Augmented Generation (RAG) Chatbot Embedded in a Published Digital Book Constitution

## Core Principles

### Accuracy and Relevance
All responses must be grounded in the book's content or user-selected text only, with no hallucinations or external knowledge leakage

### Reliability with Cohere
Use only Cohere's APIs for embeddings and text generation (absolutely no OpenAI dependencies)

### Modularity
Clean, separation-of-concerns architecture with distinct components for ingestion, retrieval, generation, and frontend integration

### User Experience
Seamless embedding in digital book formats (e.g., web-based book, PDF viewer, or custom reader) via iframe or JS widget

### Flexibility
Support two modes — (1) Full book context, (2) Isolated user-selected text only

### Security and Performance
API key protection via environment variables, rate limiting, CORS configuration, and target <4 seconds end-to-end response latency

## Technology Stack Standards
LLM Provider: Cohere (use cohere-python SDK for both embeddings and generation – models like command-r-plus or embed-english-v3.0); Backend Framework: FastAPI (async, with proper dependency injection, Pydantic models, and OpenAPI/Swagger docs); Vector Database: Qdrant Cloud Free Tier (qdrant-client library, collections with proper vector size matching Cohere embeddings); Relational Database: Neon Serverless Postgres (SQLAlchemy + asyncpg for metadata, chunk mapping, session tracking if needed); Chunking Strategy: Semantic chunking with overlap (300-500 tokens per chunk, 100-token overlap) for better retrieval quality; Retrieval: Hybrid or pure vector search with top-k=5-10, re-ranking optional using Cohere rerank endpoint if budget allows; Selected Text Mode: Dynamic temporary collection creation or in-memory vector store for user-highlighted text to ensure strict isolation

## Code Quality and Testing Standards
Code Quality: PEP 8, type hints everywhere, black/isort formatting, comprehensive logging, error handling; Testing: pytest with at least 85% coverage, including integration tests for ingestion and query pipelines; Documentation: Detailed README.md with setup, deployment, architecture diagram, and usage examples; Tech Stack Limitations: Python 3.11+, FastAPI, Uvicorn, cohere, qdrant-client, sqlalchemy/asyncpg, pydantic, langchain optional only if Cohere-compatible; Book Size: Designed to handle books up to 1,500 pages or ~1M tokens efficiently; Deployment: Containerized (Docker) for easy local/run-on-cloud setup; frontend widget in plain HTML/JS or React if needed

## Governance
No OpenAI SDK or API usage at all (even for fallback or testing); Stick strictly to free tiers: Qdrant Cloud Free, Neon Serverless Free, Cohere free tier credits where possible; All PRs/reviews must verify compliance with technology stack constraints; Complexity must be justified; Use README.md for runtime development guidance

**Version**: 1.0.0 | **Ratified**: 2025-12-17 | **Last Amended**: 2025-12-17