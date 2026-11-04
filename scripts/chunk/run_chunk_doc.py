import json
import logging
from pathlib import Path
from softmaxx.config import AppConfig
import softmaxx.boxgpt.rag as box_rag 


def main():
    AppConfig.load()
    log_config = AppConfig.get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)
    
    # @IMP: verify source, target and model paths
    SRC_DIR = Path('~/code/github/boxgpt/scripts/temp/pp').expanduser()
    SRC_FILE = "ab_ocr_pp.md"
    # verify target folder 
    TARGET_DIR = Path('~/code/github/boxgpt/scripts/temp/embedding').expanduser()
    MODEL_PATH = "/home/rjha/code/models/bge-m3",
    
    # get source 
    pp_content = SRC_DIR / SRC_FILE
    if not pp_content.exists():
        raise FileNotFoundError(f"unbale to locate pp content {pp_content}")

    # Chunk content locally
    print("Chunking processed content...")
    raw_chunks = box_rag.chunk_content(pp_content)
    print(f"Created {len(raw_chunks)} distinct chunks.")

    # Generate BGE-M3 dense embeddings
    print("Generating vector embeddings...")
    enriched_chunks = box_rag.get_embeddings(raw_chunks, model_path=MODEL_PATH)

    # Optional: Save enriched chunks to inspect 
    # payload structure before DB insertion
    
    target_path = TARGET_DIR / "doc_embedding.json"
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved embedding JSON payload to '{target_path}'.")
    box_rag.store_embedding(enriched_chunks, doc_id="TEST_DOC01")
    

    
if __name__ == "__main__":
    main()