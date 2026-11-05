                                                                    Table "public.document_chunks"
     Column     |           Type           | Collation | Nullable |                   Default                   | Storage  | Compression | Stats target | Description 
----------------+--------------------------+-----------+----------+---------------------------------------------+----------+-------------+--------------+-------------
 id             | bigint                   |           | not null | nextval('document_chunks_id_seq'::regclass) | plain    |             |              | 
 doc_id         | character varying(100)   |           | not null | 'mai_out'::character varying                | extended |             |              | 
 section_title  | text                     |           | not null |                                             | extended |             |              | 
 page_start     | integer                  |           | not null |                                             | plain    |             |              | 
 page_end       | integer                  |           | not null |                                             | plain    |             |              | 
 chunk_index    | integer                  |           | not null |                                             | plain    |             |              | 
 content        | text                     |           | not null |                                             | extended |             |              | 
 embedding_text | text                     |           | not null |                                             | extended |             |              | 
 embedding      | vector(1024)             |           | not null |                                             | external |             |              | 
 created_at     | timestamp with time zone |           |          | CURRENT_TIMESTAMP                           | plain    |             |              | 
Indexes:
    "document_chunks_pkey" PRIMARY KEY, btree (id)
    "idx_document_chunks_embedding" hnsw (embedding vector_cosine_ops)
Access method: heap

