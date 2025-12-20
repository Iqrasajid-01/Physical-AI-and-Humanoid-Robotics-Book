# Implementation Plan: RAG Chatbot for Digital Books

**Branch**: `001-rag-chatbot` | **Date**: 2025-12-17 | **Spec**: ./backend/specs/001-rag-chatbot/spec.md
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a Retrieval-Augmented Generation (RAG) chatbot that allows digital book readers to ask questions about book content and receive accurate, context-aware answers. The system includes full-book Q&A mode, selected-text mode for isolated analysis, and an embeddable widget for integration into digital book platforms. The architecture uses Cohere for embeddings and generation, Qdrant for vector storage, and Neon Postgres for metadata, all accessed through a FastAPI backend with a vanilla JavaScript frontend widget.

## Technical Context

**Language/Version**: Python 3.11, JavaScript ES6+
**Primary Dependencies**: FastAPI, Uvicorn, Cohere, Qdrant-client, SQLAlchemy, AsyncPG, Pydantic
**Storage**: Qdrant (vector database), Neon Postgres (metadata)
**Testing**: pytest with 85%+ coverage, linting (black, isort, flake8), type checking (mypy), security scanning (bandit)
**Target Platform**: Linux server deployment with web browser compatibility for frontend widget
**Project Type**: Web application with backend API and embeddable frontend widget
**Performance Goals**: <4 seconds end-to-end response time for 90% of queries, handle books up to 1,500 pages (~1M tokens)
**Constraints**: Cohere-only API usage (no OpenAI), free tier service limits (Qdrant Cloud Free, Neon Serverless Free, Cohere free tier), <4s latency targets
**Scale/Scope**: Handle individual books up to 1,500 pages, support both full-book and selected-text query modes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

1. **Accuracy and Relevance**: All responses must be grounded in book content with no hallucinations - IMPLEMENTED through strict RAG context usage
2. **Cohere-Only Dependency**: No OpenAI dependencies allowed - CONFIRMED through tech stack selection
3. **Modularity**: Clean separation between ingestion, retrieval, generation, and frontend components - IMPLEMENTED through component architecture
4. **User Experience**: Seamless embedding in digital book formats - IMPLEMENTED through embeddable widget design
5. **Mode Flexibility**: Support both full book context and isolated text modes - IMPLEMENTED through dual query modes
6. **Technical Standards**: Adherence to FastAPI, Qdrant, Neon Postgres, and Cohere stack - CONFIRMED
7. **Performance Standards**: <4 second response latency targets - TARGETED through architecture decisions
8. **Security Requirements**: API key protection, rate limiting, proper CORS - IMPLEMENTED through FastAPI middleware
9. **Code Quality**: 85%+ test coverage and PEP 8 compliance - COMMITTED through testing strategy

## Project Structure

### Documentation (this feature)

```text
backend/specs/001-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── app.py               # Main FastAPI application with routes
│   ├── models/              # Pydantic schemas for request/response objects
│   │   ├── book_content.py
│   │   ├── question.py
│   │   ├── response.py
│   │   └── query_context.py
│   ├── ingestion/           # Book content processing components
│   │   ├── chunker.py       # Text chunking with semantic awareness
│   │   ├── embedder.py      # Cohere embedding generation
│   │   └── processor.py     # Ingestion pipeline orchestrator
│   ├── retrieval/           # Vector search and document retrieval
│   │   ├── searcher.py      # Qdrant-based search implementation
│   │   └── selector.py      # Select text mode handler
│   ├── generation/          # Response generation with RAG
│   │   └── generator.py     # Cohere-based response generation
│   ├── db/                  # Database managers
│   │   ├── qdrant_manager.py # Vector database operations
│   │   └── neon_manager.py   # Metadata database operations
│   └── utils/               # Helper functions and utilities
│       ├── validators.py    # Input validation utilities
│       ├── constants.py     # Configuration constants
│       └── logger.py        # Logging utilities
├── static/                  # Static files for frontend widget
│   └── widget/
│       ├── chat-widget.css  # Styling for the embeddable widget
│       ├── chat-widget.js   # JavaScript for the embeddable widget
│       └── index.html       # Demo page for widget
├── tests/                   # Test suite
│   ├── unit/                # Unit tests for individual components
│   ├── integration/         # Integration tests for API endpoints
│   ├── contract/            # Contract tests for API compliance
│   └── e2e/                 # End-to-end tests for complete workflows
├── docs/                    # Documentation files
│   ├── architecture.md      # Architecture diagram and explanation
│   └── quickstart.md        # Quick start guide
├── requirements.txt         # Python dependencies
├── Dockerfile               # Containerization configuration
├── docker-compose.yml       # Multi-container orchestration
├── .env.example             # Environment variable template
└── README.md                # Main project documentation
```

**Structure Decision**: Web application structure selected with backend API services and embeddable frontend widget. The backend handles all processing (ingestion, retrieval, generation) while the frontend widget provides a lightweight, embeddable interface that communicates with the backend API. This supports the required modularity and enables the dual query modes while maintaining security and performance requirements.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [None] | [All constitution requirements satisfied] | [N/A] |