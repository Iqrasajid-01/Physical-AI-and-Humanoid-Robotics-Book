---
id: 001
title: RAG Chatbot Spec
stage: spec
date: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20250929
feature: rag-chatbot
branch: 001-rag-chatbot
user: user
command: /sp.specify
labels: ["rag", "chatbot", "digital-book", "specification"]
links:
  spec: specs/001-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-rag-chatbot/spec.md
  - specs/001-rag-chatbot/checklists/requirements.md
tests: []
---

## Prompt

```text
Integrated Retrieval-Augmented Generation (RAG) Chatbot Embedded in a Published Digital Book
Target audience:

Digital book authors and publishers who want to add an intelligent, interactive layer to their books
Full-stack Python developers with intermediate experience in APIs, vector databases, and async web frameworks
Readers of the final book who will interact with the chatbot for deeper understanding or quick lookups

Focus:
Build a production-ready, modular RAG system that allows readers to ask natural-language questions about the book content. The chatbot must support:

Full-book mode: Retrieve and generate answers using the entire pre-ingested book content
Selected-text mode: Dynamically embed and query only the user-highlighted/selected text, completely isolating it from the main book knowledge base
Deliver a lightweight, embeddable chat widget (HTML + JS) that can be easily integrated into any web-based digital book platform (e.g., static site, React app, or iframe in PDF viewer)

Success criteria:

Ingestion pipeline successfully chunks, embeds (using Cohere embed-english-v3.0 or similar), and stores full book text with rich metadata (page, chapter, section) in Qdrant + Neon Postgres
Query endpoint returns highly relevant, citation-backed answers for 95%+ of 50+ diverse test questions covering facts, summaries, explanations, and comparisons from the book
Selected-text mode works flawlessly: When user provides highlighted text, the system creates a temporary isolated vector collection and generates responses based solely on that text (verified by manual inspection and test cases)
Embeddable chat widget is responsive, beautiful (minimal Tailwind or vanilla CSS), and works in a demo HTML page mimicking a real digital book
End-to-end latency <4 seconds for 90% of queries on free-tier resources
Codebase is clean, fully typed (mypy clean), 85%+ test coverage, Dockerized, with comprehensive README and architecture diagram
Zero sensitive credentials hardcoded; all loaded via environment variables

Constraints:

LLM Provider: Exclusively Cohere (API Key: mHuedeSyxTxgbbKaxyjbiaRkWQGUZCASSW2z4wpo). Use cohere-python SDK for both embeddings (embed-english-v3.0 or multilingual if needed) and generation (command-r-plus preferred for reasoning quality)
Backend: FastAPI (async endpoints, dependency injection, Pydantic v2 models, automatic Swagger/ReDoc)
Vector Store: Qdrant Cloud Free Tier only
URL: https://dd851a09-90c9-49f5-badb-524e05df5a5c.europe-west3-0.gcp.cloud.qdrant.io
API Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.N9QJcVGjUHPszlyKgmKBG6b3-ScGH3dQ5FpLne-hfV0
Cluster ID: dd851a09-90c9-49f5-badb-524e05df5a5c

Relational DB: Neon Serverless Postgres only
Connection URL: postgresql://neondb_owner:npg_W52EVxmjIuRT@ep-lively-fog-agievxr9-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require

Core Python libraries: cohere, qdrant-client, fastapi, uvicorn, sqlalchemy (async), asyncpg, pydantic, python-dotenv
Optional: LangChain or LlamaIndex only if they add clear value and are fully compatible with Cohere (prefer minimal dependencies)
Book content size: Designed for books up to 1,500 pages (~1-1.5M tokens)
Deployment: Fully Dockerized with docker-compose for local development; ready for Render, Fly.io, or similar free-tier hosts

Not building:

Complex user authentication, accounts, or chat history persistence across sessions
Advanced frontend framework (no React/Vue required – keep widget lightweight)
Multi-language support beyond what Cohere naturally provides
Real-time collaborative features or analytics dashboard
Mobile-native app (focus on web-embeddable widget only)
Any integration with OpenAI, Anthropic, or other non-Cohere LLMs
```

## Response snapshot

Created a comprehensive feature specification for the RAG Chatbot for Digital Books with the following key elements:

1. **User Scenarios & Testing**:
   - User Story 1 (P1): Full-book Q&A functionality
   - User Story 2 (P2): Selected-text mode functionality
   - User Story 3 (P3): Embeddable widget integration

2. **Functional Requirements** (FR-001 to FR-010):
   - Book content ingestion and processing
   - Dual query modes (full-book and selected-text)
   - Natural language question processing
   - Proper citation and reference system
   - Context isolation between modes
   - Lightweight embeddable widget

3. **Key Entities**:
   - BookContent, Question, Response, QueryContext, and Embedding

4. **Constitution Alignment**:
   - All requirements aligned with RAG Chatbot Constitution principles

5. **Success Criteria** (SC-001 to SC-006):
   - 100% book ingestion success for 1,500 pages
   - 95%+ relevant responses with citations
   - 100% selected-text isolation
   - <4 second response time for 90% of queries
   - 95%+ widget integration success
   - 85%+ test coverage

6. **Quality Checklist**:
   - All quality criteria validated and marked complete

## Outcome

- ✅ Impact: Created comprehensive feature specification for RAG chatbot
- 🧪 Tests: No tests run yet (specification phase)
- 📁 Files: Created spec.md and requirements checklist
- 🔁 Next prompts: Ready for planning phase with /sp.plan
- 🧠 Reflection: Specification aligns with all constitutional requirements and user needs

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): All validation items pass
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin implementation planning with /sp.plan