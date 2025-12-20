# Data Model: RAG Chatbot for Digital Books

**Feature**: 001-rag-chatbot | **Date**: 2025-12-17

## Overview

This document defines the data models for the RAG chatbot system, including entity definitions, relationships, validation rules, and state transitions where applicable.

## Core Entities

### 1. BookContent
**Description**: Represents the original book text with structural metadata

**Fields**:
- `id` (UUID): Unique identifier for the book content
- `title` (str): Title of the book
- `author` (str): Author of the book
- `isbn` (str, optional): ISBN identifier
- `total_pages` (int): Total number of pages in the book
- `language` (str): Language of the book content
- `created_at` (datetime): Timestamp when content was ingested
- `updated_at` (datetime): Timestamp when content was last updated
- `metadata` (JSON): Additional book metadata (publisher, publication date, etc.)

**Validation Rules**:
- Title must be 1-200 characters
- Total pages must be > 0 and <= 1500
- Language must be a valid ISO 639-1 code
- Content must not exceed 1M tokens

### 2. ContentChunk
**Description**: Represents a semantically coherent chunk of book content for vector storage

**Fields**:
- `id` (UUID): Unique identifier for the chunk
- `book_content_id` (UUID): Reference to parent BookContent
- `chunk_text` (str): The actual text content of the chunk
- `chunk_order` (int): Order of the chunk within the book
- `start_page` (int): Starting page number of this chunk
- `end_page` (int): Ending page number of this chunk
- `start_section` (str): Starting section/chapter name
- `end_section` (str): Ending section/chapter name
- `token_count` (int): Number of tokens in the chunk
- `embedding_vector` (Vector): Cohere embedding vector (handled separately in Qdrant)
- `created_at` (datetime): Timestamp when chunk was created

**Validation Rules**:
- chunk_text must be 50-1000 tokens
- start_page must be >= 1 and <= book.total_pages
- end_page must be >= start_page
- token_count must match actual token count of chunk_text
- chunk_order must be unique within book_content_id

### 3. Question
**Description**: Represents a user's natural language query

**Fields**:
- `id` (UUID): Unique identifier for the question
- `session_id` (UUID, optional): Identifier for conversation session
- `query_text` (str): The user's question text
- `query_mode` (str): Mode of querying ('full_book' or 'selected_text')
- `selected_text` (str, optional): Text provided in selected-text mode
- `user_id` (UUID, optional): Identifier for the user (if tracking)
- `created_at` (datetime): Timestamp when question was submitted
- `processed_chunks_count` (int): Number of chunks used for response generation

**Validation Rules**:
- query_text must be 5-2000 characters
- query_mode must be either 'full_book' or 'selected_text'
- If query_mode is 'selected_text', selected_text must be provided and 10-5000 characters
- If query_mode is 'full_book', selected_text must be null

### 4. Response
**Description**: Represents the system-generated answer with citations

**Fields**:
- `id` (UUID): Unique identifier for the response
- `question_id` (UUID): Reference to the corresponding question
- `response_text` (str): The generated response text
- `confidence_score` (float): Confidence level of the response (0.0-1.0)
- `citations` (JSON): Array of citations with page/chapter references
- `retrieved_chunks_ids` (JSON): Array of chunk IDs used for generation
- `processing_time_ms` (int): Time taken to generate the response
- `generated_at` (datetime): Timestamp when response was generated
- `has_fallback_response` (bool): Whether this is a fallback response

**Validation Rules**:
- response_text must be 10-5000 characters
- confidence_score must be between 0.0 and 1.0
- citations must be a valid array of citation objects with required fields
- processing_time_ms must be >= 0

### 5. Citation
**Description**: Represents a reference to specific locations in the book

**Fields**:
- `page_numbers` (Array[int]): Page numbers referenced
- `chapter_titles` (Array[str]): Chapter titles referenced
- `section_names` (Array[str]): Section names referenced
- `snippet` (str): Relevant text snippet from the book
- `relevance_score` (float): How relevant this citation is to the response (0.0-1.0)

**Validation Rules**:
- page_numbers must be positive integers
- snippet must be 10-500 characters
- relevance_score must be between 0.0 and 1.0

### 6. QuerySession
**Description**: Represents a conversation session between user and chatbot

**Fields**:
- `id` (UUID): Unique identifier for the session
- `user_id` (UUID, optional): Identifier for the user
- `book_content_id` (UUID): Reference to the book being queried
- `created_at` (datetime): Timestamp when session started
- `last_activity_at` (datetime): Timestamp of last activity in session
- `query_count` (int): Number of queries in this session
- `is_active` (bool): Whether the session is currently active

**Validation Rules**:
- query_count must be >= 0
- last_activity_at must be >= created_at

## Relationships

### BookContent ↔ ContentChunk
- **Relationship**: One-to-Many
- **Description**: One BookContent can have many ContentChunks
- **Cardinality**: 1 book : N chunks
- **Constraint**: All chunks must belong to the same book

### Question ↔ Response
- **Relationship**: One-to-One
- **Description**: Each Question generates one Response
- **Cardinality**: 1 question : 1 response
- **Constraint**: Response must be generated for each question

### Question ↔ QuerySession
- **Relationship**: Many-to-One (optional)
- **Description**: Questions may belong to a QuerySession
- **Cardinality**: N questions : 1 session (optional)
- **Constraint**: Questions without sessions are treated as standalone

### Response ↔ ContentChunk
- **Relationship**: Many-to-Many (via retrieved_chunks_ids)
- **Description**: A Response references multiple ContentChunks
- **Cardinality**: 1 response : N chunks
- **Constraint**: Referenced chunks must exist and belong to the same book

## State Transitions

### QuerySession States
1. **CREATED**: Session initialized but no activity yet
2. **ACTIVE**: At least one query has been processed
3. **INACTIVE**: No activity for extended period (may auto-expire)
4. **CLOSED**: Session explicitly closed by user or system

**Transitions**:
- CREATED → ACTIVE: When first question is processed
- ACTIVE → INACTIVE: After configurable inactivity period
- INACTIVE → ACTIVE: When new question arrives
- ACTIVE → CLOSED: When session is explicitly closed
- INACTIVE → CLOSED: After extended inactivity

## Indexes

### BookContent
- Primary: id
- Secondary: title (for search), created_at (for ordering)

### ContentChunk
- Primary: id
- Secondary: book_content_id (for book association), chunk_order (for sequence)
- Composite: (book_content_id, chunk_order)

### Question
- Primary: id
- Secondary: session_id (for session grouping), created_at (for chronological ordering)
- Composite: (session_id, created_at)

### Response
- Primary: id
- Secondary: question_id (for question association), generated_at (for ordering)
- Composite: (question_id, generated_at)

## Constraints

1. **Data Integrity**: All foreign key references must point to existing records
2. **Size Limits**: Content chunks must not exceed token limits specified in validation rules
3. **Temporal Logic**: All timestamps must follow logical chronological order
4. **Mode Isolation**: In selected-text mode, responses must not reference main book content chunks
5. **Citation Accuracy**: All citations must correspond to actual content in referenced chunks