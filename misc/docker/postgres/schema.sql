-- Enable the vector extension

-- Enable the pgvector extension 
CREATE EXTENSION IF NOT EXISTS vector;


-- Table to keep track of source documents
CREATE TABLE IF NOT EXISTS doc_source (
    id SERIAL PRIMARY KEY,
    doc_slug VARCHAR(255) NOT NULL UNIQUE,
    doc_name VARCHAR(255)  NOT NULL,
    doc_meta JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing chunks and 1024-dim BGE-M3 embeddings
CREATE TABLE IF NOT EXISTS doc_chunks (
    id SERIAL PRIMARY KEY,
    doc_id INT REFERENCES doc_source(id) ON DELETE CASCADE,
    section_title VARCHAR(512),
    page_start INT,
    page_end INT,
    chunk_index INT,
    content TEXT NOT NULL,                -- Original text for UI display
    embedding_text TEXT NOT NULL,         -- Contextualized text used for embedding
    embedding VECTOR(1024)                -- BGE-M3 vector dimension
);

-- Create an HNSW index for fast vector similarity 
-- search using cosine distance vector operator (<=>)
CREATE INDEX IF NOT EXISTS idx_hnsw_doc_chunks 
ON doc_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- full-text search index (gin (to_tsvector(...)))
-- to implement hybrid search, combining keyword/lexical 
-- search with vector similarity search

CREATE INDEX IF NOT EXISTS idx_tsv_doc_chunks 
ON doc_chunks 
USING gin (to_tsvector('english', content));

