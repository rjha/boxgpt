import json
import logging
from pathlib import Path
from softmaxx.config import AppConfig, get_logger_config
import softmaxx.boxgpt.rag as box_rag 


def main():
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)
    
    # @IMP: verify source, target and model paths
    ROOT_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()
    SRC_FILE = ROOT_DIR / "pp" / "ab_ocr_pp.md"
    TARGET_FILE = ROOT_DIR / "chunk" / "ab_ocr_pp_chunk.json"
    MODEL_PATH = "/home/rjha/code/models/bge-m3"
    
    # get source 
    if not SRC_FILE.exists():
        raise FileNotFoundError(f"unbale to locate source {SRC_FILE}")

    # Chunk content locally
    print("Chunking processed content...")
    raw_chunks = box_rag.chunk_markdown_content(SRC_FILE)
    print(f"Created {len(raw_chunks)} distinct chunks.")

    # Generate BGE-M3 dense embeddings
    print("Generating vector embeddings...")
    enriched_chunks = box_rag.add_embeddings(raw_chunks, model_path=MODEL_PATH)

    # Optional: Save enriched chunks to inspect 
    # payload structure before DB insertion
    
    with TARGET_FILE.open("w", encoding="utf-8") as f:
        json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved embedding JSON payload to '{TARGET_FILE}'.")
    # box_rag.store_embedding(enriched_chunks, doc_id="TEST_DOC01")
    

    
if __name__ == "__main__":
    main()