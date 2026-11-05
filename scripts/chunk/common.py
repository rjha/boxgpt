import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import softmaxx.boxgpt.rag as box_rag 


def transform_slug(doc_slug: str) -> str:
    readable_text = doc_slug.replace("_", " ").replace("-", " ")
    title_str = readable_text.title()
    
    # Optional: Normalize common industrial acronyms
    acronyms = {"Lt": "LT", "Smp": "SMP", "Sop": "SOP", "Hvac": "HVAC"}
    for old, new in acronyms.items():
        title_str = title_str.replace(old, new)
        
    return title_str

def get_chunks(source_path):
    # get source 
    if not source_path.exists():
        raise FileNotFoundError(f"unable to locate source {source_path}")

    # Chunk content locally
    print("Chunking processed content...")
    raw_chunks = box_rag.chunk_markdown_content(source_path)
    print(f"Created {len(raw_chunks)} distinct chunks.")
    return raw_chunks

def add_embedding(raw_chunks, model:SentenceTransformer):
    # Generate BGE-M3 dense embeddings
    print("Generating vector embeddings...")
    enriched_chunks = box_rag.add_embeddings(chunks=raw_chunks, model=model)
    return enriched_chunks

def save_to_file(chunks, target_path):
     with target_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved embedding JSON payload to '{target_path}'.")


def save_to_pgvector(db_string, chunk_file_path, doc_info):
    if not chunk_file_path.exists():
        raise FileNotFoundError(f"unable to locate source {chunk_file_path}")

    # Load parsed JSON chunk list from file
    with chunk_file_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    # @todo Data class 
    box_rag.store_doc_chunks(
        document_slug=doc_info["slug"],
        document_name=doc_info["name"], 
        document_metadata=doc_info["meta"],
        chunks=chunks,
        conn_string=db_string)
    
    print("chunks stored in database")