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
### **Environment Configuration**:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key  # For web search functionality
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

## **4. Enhanced Document Processing**

- **Multi-format Support**: Uses `UnstructuredLoader` to handle various document types (PDF, DOCX, TXT)
- **Optimized Chunking**: `RecursiveCharacterTextSplitter` with 1000 character chunks and 150 character overlap
- **Batch Processing**: Efficient ingestion of multiple documents from data folder
- **Duplicate Prevention**: Hash-based deduplication prevents redundant embeddings

---

## **5. Implemented Hashing and comparison to avoid duplicate vector embeddings**

- Used SHA256 hashing mechanism to hash the content
- Compared and skipped duplicate content

```python
def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
```

- Updated database schema to include `content_hash` column for efficient duplicate detection
- Modified insertion logic in `vector_store.py` to check existing hashes before inserting new documents

---

## **5. Context Retrieval System Implementation**

- Created `retrieve.py` module to handle semantic search and context retrieval:

  - **Query Embedding**: Convert user queries to embeddings using `text-embedding-3-large`
  - **Vector Similarity Search**: Use Supabase RPC function `match_documents` for efficient cosine similarity search
  - **Context Aggregation**: Combine top-k most relevant document chunks into coherent context

- **Key Features**:
  ```python
  def retrieve_context(query, k=3):
      # Generate query embedding
      query_embedding = embeddings.embed_query(query)

      # Perform similarity search via RPC
      result = supabase.rpc("match_documents", {
          "match_count": k,
          "query_embedding": query_embedding
      }).execute()

      # Return aggregated context
      return "\n\n".join(documents)
  ```

---

## **6. Intelligent Chatbot with Hybrid RAG-Web Search**

- Implemented `chatbot.py` using **LangGraph** for sophisticated conversation flow
- **Multi-tiered Response Strategy**:
  1. **RAG Context Retrieval**: First attempt to find relevant information from vector database
  2. **LLM General Knowledge**: Fall back to model's built-in knowledge if RAG insufficient
  3. **Web Search Trigger**: Only search internet if LLM explicitly indicates need with "Should search internet"
  4. **Comprehensive Final Answer**: Combine multiple information sources intelligently

### **Chatbot Architecture**:

```
User Query → RAG Retrieval → LLM Processing → Search Decision → Final Response
```

### **Key Components**:

- **State Management**: Single `requires_search` boolean flag for efficient flow control
- **OpenAI Integration**: GPT-4o-mini for natural language processing and decision making
- **Web Search**: Google Serper API for real-time information when needed
- **Smart Fallback**: Graceful degradation from RAG → General Knowledge → Web Search

### **Flow Optimization**:

- **Minimized API Calls**: Only 1 LLM call per query unless search needed
- **Intelligent Search Triggering**: Exact phrase matching prevents unnecessary web searches
- **Context Combination**: Seamlessly merges RAG context with search results when both available

---



## **7. Complete Application Interface**

- Created `app.py` as main entry point with two operational modes:
  1. **Ingestion Mode**: Process and store new documents in vector database
  2. **Chatbot Mode**: Interactive conversation with RAG and web search capabilities



---


## **8. Project Structure**

```
ChatBot/
├── app.py                      # Main application entry point
├── chatbot.py                  # LangGraph-based conversation flow
├── retrieve.py                 # RAG context retrieval system
├── ingest.py                   # Document processing and ingestion
├── vector_store.py             # Supabase vector database operations
├── utils.py                    # Utility functions (hashing, document loading)
├── requirements.txt            # Python dependencies
├── example.env                 # Environment configuration template
├── data/                       # Document storage folder
├── FLOW_SUMMARY.md            # Detailed flow documentation
└── test_*.py                  # Testing suite
```

---

## **9. Usage Instructions**

### **Setup**:

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables in `.env` file
3. Ensure Supabase database is set up with required table and RPC function

### **Running the Application**:

```bash
# Process documents into vector database
python app.py  # Select option 1

# Start interactive chatbot
python app.py  # Select option 2

# Or run chatbot directly
python chatbot.py
```

### **Adding New Documents**:

1. Place documents in `./data/` folder
2. Run ingestion mode to process and store embeddings
3. New documents are automatically integrated into RAG system
