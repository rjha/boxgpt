import logging
from pathlib import Path
import softmaxx.config as s_config
import chunk.common as chunker 


def main():
    s_config.AppConfig.load()
    log_config = s_config.get_logger_config("local")
    s_config.AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)
    
    # @IMP: verify source, target and model paths
    BUILD_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()
    DOC_NAME = "ab_ocr_pp.md"
    SRC_PATH = BUILD_DIR / "pp" / DOC_NAME
    MODEL_PATH = "/home/rjha/code/models/bge-m3"
    TARGET_PATH = BUILD_DIR / "chunk" / "ab_ocr_pp_chunk.json"
    
    # step by step verification
    raw_chunks = chunker.get_chunks(SRC_PATH)
    enriched_chunks = chunker.add_embedding(raw_chunks=raw_chunks, model_path=MODEL_PATH)
    chunker.save_to_file(chunks=enriched_chunks, target_path=TARGET_PATH)
    
    db_config:s_config.DatabaseConfig = s_config.get_database_config()
    db_string = db_config.get_uri() 
    
    chunker.save_to_pgvector(
        db_string=db_string,
        chunk_file_path=TARGET_PATH, 
        doc_info={
            "name": DOC_NAME,
            "meta": {
                "source_file": DOC_NAME,
                "pipeline": "bge-m3-rag",
            }
        }
    )
    
    
    
if __name__ == "__main__":
    main()