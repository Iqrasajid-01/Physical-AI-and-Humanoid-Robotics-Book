---
id: 001
title: RAG Chatbot Constitution
stage: constitution
date: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: user
command: /sp.constitution
labels: ["constitution", "rag", "chatbot", "governance"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
  - .specify/templates/plan-template.md
  - .specify/templates/spec-template.md
  - .specify/templates/tasks-template.md
tests: []
---

## Prompt

```text
Integrated Retrieval-Augmented Generation (RAG) Chatbot Embedded in a Published Digital Book
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
Response Latency: Target <4 seconds end-to-end for typical querie
```

## Response snapshot

Created a comprehensive constitution for the Integrated Retrieval-Augmented Generation (RAG) Chatbot project with specific principles and guidelines. The constitution includes:

1. Core Principles:
   - Accuracy and Relevance: Ensures responses are grounded in book content with no hallucinations
   - Cohere-Only Dependency: Prohibits any OpenAI dependencies
   - Modularity and Separation of Concerns: Clean architecture between components
   - User Experience Excellence: Seamless embedding in digital book formats
   - Mode Flexibility: Support for both full book context and isolated text modes

2. Technical Standards:
   - Technology stack requirements (FastAPI, Qdrant, Neon Postgres, Cohere)
   - Performance standards (response latency <4 seconds)
   - Security requirements (API key protection, rate limiting)
   - Code quality standards (PEP 8, 85% test coverage)

3. Development Workflow:
   - Testing requirements (unit, integration, end-to-end)
   - Documentation standards
   - Deployment policies

4. Governance:
   - Amendment procedures
   - Compliance requirements

Updated dependent templates to align with the new constitution:
- plan-template.md: Added Constitution Check section
- spec-template.md: Added Constitution Alignment section
- tasks-template.md: Added Constitution Compliance section

## Outcome

- ✅ Impact: Created comprehensive governance framework for RAG chatbot project
- 🧪 Tests: No tests run
- 📁 Files: Updated constitution and 3 template files
- 🔁 Next prompts: Plan and implement RAG chatbot features
- 🧠 Reflection: Constitution provides clear architectural and implementation guidance

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin feature specification based on constitution