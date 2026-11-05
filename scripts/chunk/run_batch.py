import signal 
import sys
import time
import logging 
from pathlib import Path
import softmaxx.config as s_config
import chunk.common as chunker 


def signal_handler(sig, frame):
    print("\n[!] aborting on keyboard press...")
    sys.exit(130)


def process_batch(source_dir: Path, target_dir: Path, model_path:str):
    logger = logging.getLogger("main." + __name__)
    # input check: source_dir and target_dir
    if not (source_dir.exists() and source_dir.is_dir()):
        raise ValueError("source directory not found")

    if not (target_dir.exists() and target_dir.is_dir()):
        raise ValueError("target directory not found")

    
    # gather all source files from source_dir
    md_files = sorted(list(source_dir.rglob("*.json")))
    
    for md_file in md_files:
        print(f"markdown content file: {md_file}")

        # Target is where chunk+embeddings will be saved
        doc_name = md_file.stem 
        target_path = target_dir / f"{doc_name}_chunk.json"
        print(f"source doc: {doc_name}, target_path {target_path}")

        raw_chunks = chunker.get_chunks(md_file)
        enriched_chunks = chunker.add_embedding(raw_chunks=raw_chunks, model_path=model_path)
        chunker.save_to_file(chunks=enriched_chunks, target_path=target_path)

        # Run database operation using chunk+embeddings file 
        db_config:s_config.DatabaseConfig = s_config.get_database_config()
        db_string = db_config.get_uri() 
        chunker.save_to_pgvector(
            db_string=db_string,
            chunk_file_path=target_path, 
            doc_info={
                "name": doc_name,
                "meta": {
                    "source_file": doc_name,
                    "pipeline": "bge-m3-rag",
                }
            }
        )
        


                
def main():
   
    print(" ##################################### ")
    print(" @@@  CHUNKS BATCH SCRIPT   @@@        ")
    print(" @@@  Press CTRL+C to abort        @@@ ")
    print(" ##################################### ")
    
    time.sleep(1)

    s_config.AppConfig.load()
    log_config = s_config.get_logger_config("local")
    s_config.AppConfig.init_logging(log_file=log_config.log_file)

    # @IMP: check SRC and TARGET_DIR
    SRC_DIR = Path('~/backup/ab-out/smp/pp').expanduser()
    MODEL_PATH = "/home/rjha/code/models/bge-m3"
    TARGET_DIR = Path('~/code/github/boxgpt/scripts/x.out/chunks').expanduser()

    # run the batch
    process_batch(source_dir=SRC_DIR, target_dir=TARGET_DIR, model_path=MODEL_PATH)



if __name__ == "__main__":
    # Register the SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)
    main()
