import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import softmaxx.config as s_config
import scripts.chunk.common as chunker 


def main():
    s_config.AppConfig.load()
    log_config = s_config.get_logger_config("local")
    s_config.AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)
    
    # @IMP: verify source, target and model paths
    BUILD_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()
    BACKUP_DIR = Path('/home/rjha/backup/ab-out/smp/pp')
    FILE_NAME = "smpe_i018_maintenance_procedure_for_lt_breakers.md"
    SRC_PATH = BACKUP_DIR / FILE_NAME
    
    MODEL_PATH = "/home/rjha/code/models/bge-m3"
    TARGET_PATH = BUILD_DIR / "chunk" / "ab_pp_chunk.json"
    
    # step by step verification
    model_instance = SentenceTransformer(MODEL_PATH)
    print(f" loaded model from {MODEL_PATH}")

    raw_chunks = chunker.get_chunks(SRC_PATH)
    enriched_chunks = chunker.add_embedding(raw_chunks=raw_chunks, model=model_instance)
    chunker.save_to_file(chunks=enriched_chunks, target_path=TARGET_PATH)
    
    db_config:s_config.DatabaseConfig = s_config.get_database_config()
    db_string = db_config.get_uri() 

    doc_slug = SRC_PATH.stem 
    doc_name = chunker.transform_slug(doc_slug)

    
    chunker.save_to_pgvector(
        db_string=db_string,
        chunk_file_path=TARGET_PATH, 
        doc_info={
            "slug": doc_slug,
            "name": doc_name,
            "meta": {
                "source_file": SRC_PATH.name,
                "pipeline": "bge-m3-rag",
            }
        }
    )
    
    
    
if __name__ == "__main__":
    main()