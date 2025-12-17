---
id: 1
title: RAG Chatbot Constitution
stage: constitution
date: 2025-12-17
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: main
user: Claude User
command: /sp.constitution
labels: ["constitution", "rag", "chatbot"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - null
---

## Prompt

```text
/sp.constitution
Project: Integrated Retrieval-Augmented Generation (RAG) Chatbot Embedded in a Published Digital Book
Core principles:

Accuracy and relevance: All responses must be grounded in the book's content or user-selected text only, with no hallucinations or external knowledge leakage
Reliability: Use only Cohere's APIs for embeddings and text generation (absolutely no OpenAI dependencies)
Modularity: Clean, separation-of-concerns architecture with distinct components for ingestion, retrieval, generation, and frontend integration
User experience: Seamless embedding in digital book formats (e.g., web-based book, PDF viewer, or custom reader) via iframe or JS widget
Flexibility: Support two modes — (1) Full book context, (2) Isolated user-selected text only

Key standards:

LLM Provider: Cohere (use cohere-python SDK for both embeddings and generation – models like command-r-plus or embed-english-v3.0)
Backend Framework: FastAPI (async, with proper dependency injection, Pydantic models, and OpenAPI/Swagger docs)
Vector Database: Qdrant Cloud Free Tier (qdrant-client library, collections with proper vector size matching Cohere embeddings)
Relational Database: Neon Serverless Postgres (SQLAlchemy + asyncpg for metadata, chunk mapping, session tracking if needed)
Chunking Strategy: Semantic chunking with overlap (300-500 tokens per chunk, 100-token overlap) for better retrieval quality
Retrieval: Hybrid or pure vector search with top-k=5-10, re-ranking optional using Cohere rerank endpoint if budget allows
Selected Text Mode: Dynamic temporary collection creation or in-memory vector store for user-highlighted text to ensure strict isolation
Security: API key protection via environment variables, rate limiting, CORS configuration
Code Quality: PEP 8, type hints everywhere, black/isort formatting, comprehensive logging, error handling
Testing: pytest with at least 85% coverage, including integration tests for ingestion and query pipelines
Documentation: Detailed README.md with setup, deployment, architecture diagram, and usage examples

Constraints:

No OpenAI SDK or API usage at all (even for fallback or testing)
Stick strictly to free tiers: Qdrant Cloud Free, Neon Serverless Free, Cohere free tier credits where possible
Tech Stack Limitations: Python 3.11+, FastAPI, Uvicorn, cohere, qdrant-client, sqlalchemy/asyncpg, pydantic, langchain optional only if Cohere-compatible
Book Size: Designed to handle books up to 1,500 pages or ~1M tokens efficiently
Response Latency: Target <4 seconds end-to-end for typical queries on free-tier resources
Deployment: Containerized (Docker) for easy local/run-on-cloud setup; frontend widget in plain HTML/JS or React if needed

Success criteria:

Ingestion pipeline successfully processes full book text and stores embeddings in Qdrant with verifiable metadata linking back to pages/chapters
Chatbot answers 95%+ of 50+ manually crafted book-specific test questions correctly and relevantly (grounded in retrieved context)
Selected-text mode works perfectly: Responses use only the provided text, confirmed by zero retrieval from main book collection
Embeddable widget demo: Fully functional chat interface embedded in a sample HTML book page with real-time interaction
Code passes linting, type checking (mypy), and security scan (bandit)
Zero critical bugs in production-like testing; graceful degradation on rate limits or errors
Complete project structure ready for deployment with clear instructions
```

## Response snapshot

Updated the project constitution file (.specify/memory/constitution.md) with the specific requirements for the Integrated Retrieval-Augmented Generation (RAG) Chatbot project. The constitution now includes:

- Core principles: Accuracy and Relevance, Reliability with Cohere, Modularity, User Experience, Flexibility, Security and Performance
- Technology Stack Standards: Details about using Cohere, FastAPI, Qdrant, Neon Postgres, and other specified technologies
- Code Quality and Testing Standards: Requirements for PEP 8, type hints, testing coverage, etc.
- Governance: Constraints about no OpenAI usage, free tier adherence, and compliance requirements
- Version 1.0.0 with ratification date 2025-12-17

## Outcome

- ✅ Impact: Created complete project constitution that will guide all development decisions for the RAG Chatbot project
- 🧪 Tests: N/A (documentation update)
- 📁 Files: Updated .specify/memory/constitution.md with new project principles
- 🔁 Next prompts: Ready to proceed with project planning and implementation
- 🧠 Reflection: Constitution provides clear boundaries and technical constraints that will ensure the project stays aligned with requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin project planning based on the established constitution