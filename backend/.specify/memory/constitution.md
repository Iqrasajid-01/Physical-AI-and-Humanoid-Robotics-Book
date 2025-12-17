<!--
Sync Impact Report:
- Version change: N/A -> 1.0.0
- Modified principles: Created new principles based on RAG chatbot requirements
- Added sections: Core principles specific to RAG chatbot, Technical Standards, Development Workflow
- Removed sections: N/A (new constitution)
- Templates requiring updates: ✅ Updated
- Follow-up TODOs: None
-->
# Integrated Retrieval-Augmented Generation (RAG) Chatbot Constitution

## Core Principles

### Accuracy and Relevance
All responses must be grounded in the book's content or user-selected text only, with no hallucinations or external knowledge leakage. The system MUST verify that generated responses are factually consistent with retrieved context and MUST NOT fabricate information or use external knowledge sources.

### Cohere-Only Dependency
The system MUST use only Cohere's APIs for embeddings and text generation with absolutely no OpenAI dependencies. All LLM interactions MUST utilize cohere-python SDK with specified models like command-r-plus or embed-english-v3.0. Any alternative providers are strictly prohibited without explicit constitutional amendment.

### Modularity and Separation of Concerns
The architecture MUST maintain clean separation between ingestion, retrieval, generation, and frontend integration components. Each module MUST be independently testable and maintainable, with well-defined interfaces and minimal coupling between components.

### User Experience Excellence
The system MUST provide seamless embedding in digital book formats via iframe or JS widget. Integration MUST be unobtrusive and enhance the reading experience without disrupting the natural flow of book consumption.

### Mode Flexibility
The system MUST support two distinct operational modes: (1) Full book context retrieval and (2) Isolated user-selected text only. Mode selection MUST be dynamic and clearly communicated to users.

## Technical Standards

### Technology Stack Requirements
- Backend: FastAPI (async, with proper dependency injection, Pydantic models, and OpenAPI/Swagger docs)
- Vector Database: Qdrant Cloud Free Tier (qdrant-client library)
- Relational Database: Neon Serverless Postgres (SQLAlchemy + asyncpg)
- LLM Provider: Cohere only (cohere-python SDK)
- Chunking: Semantic chunking with 300-500 tokens per chunk, 100-token overlap
- Retrieval: Hybrid or pure vector search with top-k=5-10

### Performance Standards
- Response Latency: Target <4 seconds end-to-end for typical queries
- Book Size: Designed to handle books up to 1,500 pages or ~1M tokens efficiently
- Concurrent Sessions: Support reasonable load based on free-tier limitations

### Security Requirements
- API keys MUST be protected via environment variables only
- Rate limiting MUST be implemented to prevent abuse
- CORS configuration MUST be properly configured for embedded widgets
- No sensitive data MUST be logged or stored unnecessarily

### Code Quality Standards
- PEP 8 compliance with type hints everywhere
- Black/isort formatting enforced
- Comprehensive logging with structured format
- Proper error handling with graceful degradation
- At least 85% test coverage with pytest

## Development Workflow

### Testing Requirements
- Unit tests for all core functions with 85%+ coverage
- Integration tests for ingestion and query pipelines
- End-to-end tests for the complete RAG flow
- Performance tests to ensure latency targets are met

### Documentation Standards
- Detailed README.md with setup, deployment, architecture diagram, and usage examples
- Inline documentation for public APIs and complex algorithms
- Architecture decision records for significant technical choices
- User guides for different operational modes

### Deployment Policies
- Strict adherence to free tier limitations: Qdrant Cloud Free, Neon Serverless Free, Cohere free tier credits
- Environment-specific configurations for development, staging, and production
- Automated testing before any deployment
- Rollback procedures documented and tested

## Governance

This constitution governs all development activities for the RAG chatbot project. All code reviews, architectural decisions, and feature implementations MUST comply with these principles. Any deviation requires constitutional amendment following the established governance process.

Amendments to this constitution require:
1. Clear justification for the change
2. Impact assessment on existing codebase
3. Approval from project maintainers
4. Update to all dependent artifacts (templates, specs, plans, tasks)

**Version**: 1.0.0 | **Ratified**: 2025-12-17 | **Last Amended**: 2025-12-17