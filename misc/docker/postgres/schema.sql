-- Enable the vector extension


-- Create table to store Hindi metadata & embeddings
CREATE TABLE test_docs (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(100),
    section_title TEXT,
    content TEXT,
    metadata JSONB,
    -- BAAI/bge-m3 output dimension is 1024
    embedding vector(1024)
);

-- Create an HNSW index for fast Cosine Similarity search
CREATE INDEX ON test_docs 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);


-- GIN index for fast keyword/full-text matching
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_tsv 
ON document_chunks 
USING gin (to_tsvector('hindi', content));

