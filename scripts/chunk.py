import json
import psycopg2
import logging

from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from softmaxx.config import get_database_config, DatabaseConfig
from softmaxx.config import AppConfig, get_logger_config

logger = logging.getLogger("main." + __name__)

def chunk_content(
    json_path: Path, 
    max_chunk_chars: int = 1500, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Reads structured section JSON and produces a list of chunk dictionaries
    with section context prepended to the embedding text.
    """
    with json_path.open("r", encoding="utf-8") as f:
        sections = json.load(f)

    # Recursive splitter for sections exceeding max_chunk_chars
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "। ", ". ", " ", ""]  # Includes Devanagari full stop '।'
    )

    chunks: List[Dict[str, Any]] = []

    for section in sections:
        section_title = section.get("section_title", "General")
        page_range = section.get("page_range", [1, 1])
        raw_content = section.get("content", "").strip()

        if not raw_content:
            continue

        # If section is small enough, keep as a single chunk
        if len(raw_content) <= max_chunk_chars:
            section_chunks = [raw_content]
        else:
            # Recursively split large sections into sub-chunks
            section_chunks = text_splitter.split_text(raw_content)

        for idx, chunk_text in enumerate(section_chunks):
            # Context Prepending: Crucial for vector search quality
            embedding_text = f"[भाग {section_title}]\n\n{chunk_text}"

            chunks.append({
                "section_title": section_title,
                "page_start": page_range[0],
                "page_end": page_range[1],
                "chunk_index": idx,
                "content": chunk_text,                # Original text (for UI display)
                "embedding_text": embedding_text,     # Contextualized text (for BGE-M3)
                "embedding": []                       # To be populated by get_chunk_embeddings
            })

    return chunks


def get_chunk_embeddings(
    chunks: List[Dict[str, Any]], 
    model_path: str = "./models/bge-m3", 
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Encodes chunk texts in batches using BAAI/bge-m3 and attaches 
    1024-dimensional float arrays to each chunk object.
    """
    if not chunks:
        return []

    print(f"Loading transformer model from: {model_path}...")
    model = SentenceTransformer(model_path)

    # Extract all prepended texts into a flat list for batching
    texts_to_encode = [chunk["embedding_text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts_to_encode)} chunks (Batch size: {batch_size})...")
    
    # Generate embeddings in a single batch pass (returns NumPy array)
    embeddings = model.encode(
        texts_to_encode, 
        batch_size=batch_size, 
        show_progress_bar=True,
        normalize_embeddings=True  # Normalizing simplifies cosine distance calculations
    )

    # Convert NumPy arrays to Python float lists and attach back to payload dicts
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def store_in_database(
    chunks: List[Dict[str, Any]], 
    doc_id: str
) -> None:
    """
    Batch inserts enriched chunks (metadata, text, and bge-m3 embeddings) 
    into PostgreSQL with pgvector.
    """
    if not chunks:
        print("No chunks provided to store.")
        return

    print(f"Connecting to PostgreSQL database...")
    db_config:DatabaseConfig = get_database_config()
    print(f"Connecting to PostgreSQL {db_config}...")
    conn = psycopg2.connect(**db_config.get_map())
    
    try:
        # Register pgvector extension handler for psycopg2
        register_vector(conn)
        cursor = conn.cursor()

        # Prepare tuple records matching table schema
        insert_records = []
        for chunk in chunks:
            insert_records.append((
                doc_id,
                chunk["section_title"],
                chunk["page_start"],
                chunk["page_end"],
                chunk["chunk_index"],
                chunk["content"],
                chunk["embedding_text"],
                chunk["embedding"]  # pgvector converts Python list[float] to vector automatically
            ))

        sql_query = """
            INSERT INTO document_chunks (
                doc_id,
                section_title,
                page_start,
                page_end,
                chunk_index,
                content,
                embedding_text,
                embedding
            ) VALUES %s;
        """

        print(f"Batch inserting {len(insert_records)} chunks into PostgreSQL...")
        
        # Execute batch insertion efficiently
        execute_values(cursor, sql_query, insert_records)
        
        # Commit transaction
        conn.commit()
        print(f"Successfully stored {len(insert_records)} chunks in database!")

    except Exception as e:
        conn.rollback()
        print(f"Error inserting chunks into database: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()

def process():
    local_config = {
        "preview": False, 
        "doc_id": "ICAR_MAG_JAN2026",
        "model_path": "/home/rjha/code/models/bge-m3",
        "content_file": "./doc01_ocr_content.json"
    }
    
    ocr_content_path = Path(local_config["content_file"])
    if not ocr_content_path.exists():
        raise FileNotFoundError(f"Could not find {ocr_content_path}")

    # Step 1: Chunk content locally
    print("Step 1: Chunking structured OCR content...")
    raw_chunks = chunk_content(ocr_content_path)
    print(f"Created {len(raw_chunks)} distinct chunks.")

    # Step 2: Generate BGE-M3 dense embeddings
    print("Step 2: Generating vector embeddings...")
    enriched_chunks = get_chunk_embeddings(raw_chunks, model_path=local_config["model_path"])

    # Optional: Save enriched chunks to inspect 
    # payload structure before DB insertion
    if local_config["preview"]:
        output_preview = Path("chunks_with_embeddings.json")
        with output_preview.open("w", encoding="utf-8") as f:
            json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved embedding JSON payload to '{output_preview}'.")
        
    
    # Step 3: Store directly into PostgreSQL
    print("Step 3: Storing vectors in PostgreSQL...")
    store_in_database(enriched_chunks, doc_id=local_config["doc_id"])
    print("\nIngestion pipeline executed successfully!")


def main():
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)
    process()

    
if __name__ == "__main__":
    main()