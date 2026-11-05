import json
import logging
from pathlib import Path
import softmaxx.config as s_config
import softmaxx.boxgpt.rag as box_rag 


def get_raw_chunks(source_path):
    # get source 
    if not source_path.exists():
        raise FileNotFoundError(f"unable to locate source {source_path}")

    # Chunk content locally
    print("Chunking processed content...")
    raw_chunks = box_rag.chunk_markdown_content(source_path)
    print(f"Created {len(raw_chunks)} distinct chunks.")
    return raw_chunks

def add_embedding_to_chunks(raw_chunks, model_path:str):
    # Generate BGE-M3 dense embeddings
    print("Generating vector embeddings...")
    enriched_chunks = box_rag.add_embeddings(raw_chunks, model_path=model_path)
    return enriched_chunks

def save_chunks_to_file(chunks, target_path):
     with target_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved embedding JSON payload to '{target_path}'.")


def save_chunks_file_to_pgvector(chunk_file_path, document_name):
    if not chunk_file_path.exists():
        raise FileNotFoundError(f"unable to locate source {chunk_file_path}")

    # Load parsed JSON chunk list from file
    with chunk_file_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Define proper JSONB metadata dictionary
    doc_metadata = {
        "source_file": document_name,
        "pipeline": "bge-m3-rag"
    }

    db_config:s_config.DatabaseConfig = s_config.get_database_config()
    box_rag.store_doc_chunks(
        document_name=document_name, 
        document_metadata=doc_metadata,
        chunks=chunks,
        conn_string=db_config.get_uri())
    
    print("chunks stored in database")

def main():
    s_config.AppConfig.load()
    log_config = s_config.get_logger_config("local")
    s_config.AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)
    
    # @IMP: verify source, target and model paths
    BUILD_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()
    SRC_DOC_NAME = "ab_ocr_pp.md"
    SRC_PATH = BUILD_DIR / "pp" / SRC_DOC_NAME
    TARGET_PATH = BUILD_DIR / "chunk" / "ab_ocr_pp_chunk.json"
    MODEL_PATH = "/home/rjha/code/models/bge-m3"

    # step by step verification
    # raw_chunks = get_raw_chunks(SRC_PATH)
    # enriched_chunks = add_embedding_to_chunks(raw_chunks=raw_chunks, model_path=MODEL_PATH)
    # save_chunks_to_file(target_path=TARGET_PATH, chunks=enriched_chunks)
    save_chunks_file_to_pgvector(chunk_file_path=TARGET_PATH, document_name=SRC_DOC_NAME)
    
    

    
if __name__ == "__main__":
    main()