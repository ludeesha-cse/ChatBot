# RAG Vector Database Project – Progress Documentation

This document summarizes the work completed so far on building a **RAG pipeline using Supabase vector database and OpenAI embeddings (text-embedding-3-large)**.

---

## **1. Environment Setup**

- Created a Python virtual environment `myenv`.
- Installed necessary packages:
  - `langchain`
  - `langchain-openai`
  - `supabase`
  - `python-dotenv`
  - `psycopg2`
- Created a `.env` file with the following variables:
  ```env
  SUPABASE_URL=https://your-project.supabase.co
  SUPABASE_SERVICE_KEY=your_service_role_key
  OPENAI_API_KEY=your_openai_api_key
  ```

## **2. Supabase Database Setup**

- Created a Supabase project.
- Created a table documents with the following columns:

  - id (UUID, primary key)
  - content (text)
  - embedding (vector, 3072 dimensions for text-embedding-3-large)
  - metadata (jsonb)

- Database Table creation
    ```
    CREATE EXTENSION IF NOT EXISTS vector;

    create table if not exists documents (
        id uuid primary key,
        content text,
        embedding vector(3072),
        metadata jsonb,
        content_hash text unique
    );
    ```
- Database RPC function

    ```
    create or replace function public.match_documents(
    query_embedding vector(3072),
    match_count int
    )
    returns table(
        id uuid,
        content text,
        metadata jsonb,
        similarity float
    )
    language sql
    as $$
        select
            id,
            content,
            metadata,
            1 - (embedding <=> query_embedding) as similarity
        from documents
        order by embedding <=> query_embedding
        limit match_count;
    $$;

    ```

- Enabled pgvector extension for vector storage.
- Attempted retrieval via PostgREST client, but realized that vector similarity requires a stored procedure for correct RPC call.

## **3. Ingestion Pipeline Implementation**

- Created ingest.py to handle:

  - Loading documents using TextLoader.
  - Chunking text using RecursiveCharacterTextSplitter with chunk_size=1000 and chunk_overlap=150.
  - Generating embeddings using OpenAIEmbeddings with text-embedding-3-large (3072 dimensions).
  - Inserting chunks into Supabase via supabase.table("documents").insert(...).

- Modularized Supabase connection into vector_store.py with:

  - init_supabase() → initializes the Supabase client.
  - insert_documents() → inserts chunks with embeddings and metadata.

- Verified ingestion works, chunks stored in Supabase with embeddings.

## **4. Implemented Hashing and comparsion to avoid duplicate vector embeddings**

- Used SHA256 hashing mechnaism to hash the context
- Compared and skipped duplicate context 
```
  def compute_hash(text: str) -> str:
      """Compute SHA256 hash of text."""
      return hashlib.sha256(text.encode('utf-8')).hexdigest()
```

