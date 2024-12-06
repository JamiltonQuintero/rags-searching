# RAG Search Engine

A sophisticated search engine implementation leveraging multiple RAG (Retrieval-Augmented Generation) approaches for enhanced document retrieval and querying.

## Overview

This project implements three cutting-edge RAG strategies:

### 1. Contextual RAG (Anthropic Approach)
Based on Anthropic's research, this implementation:
- Preserves broader document context during retrieval
- Uses a two-stage retrieval process
- Generates contextual explanations for each chunk
- Implements caching with Portkey.ai to optimize LLM calls

https://www.anthropic.com/news/contextual-retrieval

### 2. Late Chunking (Jina AI Approach)
Following Jina AI's methodology:
- Delays text segmentation until after embedding
- Improves semantic coherence in long documents
- Reduces information loss during chunking
- Optimizes token usage for large documents

https://jina.ai/news/late-chunking-in-long-context-embedding-models/

### 3. Hybrid Search (Elasticsearch)
Combines multiple search strategies:
- Vector search for semantic understanding
- Keyword search for precise matching
- BM25 scoring for relevance ranking
- Configurable weights between search types

https://www.elastic.co/search-labs/tutorials/search-tutorial/vector-search/hybrid-search

## Features

- **Multiple Processing Strategies**:
  - Basic RAG with naive chunking
  - Context-aware RAG with semantic preservation
  - Late chunking for improved segmentation
  - Hybrid search combining multiple approaches

- **Advanced Retrieval**:
  - Comparative search across implementations
  - Context-aware document understanding
  - Semantic similarity matching
  - Optimized caching for LLM operations

## Architecture

### Core Components
- FastAPI backend
- PostgreSQL with pgvector for vector storage
- Elasticsearch for hybrid search
- Multiple embedding models:
  - OpenAI embeddings
  - Google AI embeddings
  - Jina AI embeddings

### Text Processing Pipeline
1. Document ingestion (PDF support)
2. Text extraction and cleaning
3. Semantic chunking
4. Embedding generation
5. Vector storage and indexing

## API Endpoints

### Document Processing
- `POST /upload-pdf`: Main upload endpoint with optimized processing
- `POST /upload-pdf/naive`: Basic RAG implementation
- `POST /upload-pdf/contextual`: Contextual RAG implementation
- `POST /upload-pdf/jina`: Late chunking implementation
- `POST /upload-pdf/elasticsearch`: Hybrid search implementation

### Search
- `POST /query/naive`: Basic vector search
- `POST /query/contextual`: Context-aware search
- `POST /query/jina`: Late chunking search
- `POST /query/hybrid-search`: Combined vector and keyword search
- `POST /query/comparative`: Comparative results across implementations

Open Ia Caching
https://portkey.ai/docs/integrations/llms/openai/prompt-caching-openai

## Setup

1. Clone the repository
2. Install dependencies:

POSTGRES_CONNECTION_STRING=your_connection_string
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
PORTKEY_API_KEY=your_portkey_key
JINA_API_KEY=your_jina_key
ES_URL=your_elasticsearch_url (cloud)

Set up your own elastick searck service with docker
https://python.langchain.com/docs/integrations/vectorstores/elasticsearch/