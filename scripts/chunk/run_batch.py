import signal 
import sys
import time
import logging 
from pathlib import Path
import softmaxx.boxgpt.rag as box_rag 
from softmaxx.boxgpt.client.llm import DigitalOceanClient 
from scripts.constants import XOUT_DIR, DOC_DIR

from softmaxx.config import (
    AppConfig, 
    get_logger_config, 
    get_database_config,
    DatabaseConfig
)

from softmaxx.boxgpt.encoder import (
    TextEncoder,
    LocalBGEM3Encoder, 
    DOBGEM3Encoder
)

from scripts.chunk.common import (
    read_json_file,
    save_json_to_file,
    transform_slug
)


def signal_handler(sig, frame):
    print("\n[!] aborting on keyboard press...")
    sys.exit(130)


def process_batch(source_dir: Path, target_dir: Path, text_encoder:TextEncoder):
    logger = logging.getLogger("main." + __name__)
    # input check: source_dir and target_dir
    if not (source_dir.exists() and source_dir.is_dir()):
        raise ValueError("source directory not found")

    if not (target_dir.exists() and target_dir.is_dir()):
        raise ValueError("target directory not found")

    
    # gather all source files from source_dir
    # @IMP: check extension
    md_files = sorted(list(source_dir.rglob("*.md")))
    
    for md_file in md_files:
        logger.info(f"process markdown file: {md_file}")

        # chunk+embeddings will be saved to target dir
        doc_slug = md_file.stem 
        doc_name = transform_slug(doc_slug=doc_slug)
        logger.info(f"doc slug:{doc_slug}, name: {doc_name}")
        target_path = target_dir / f"{doc_slug}_chunk.json"

        # skip if already processed
        if target_path.exists():
            logger.info(f"skipping chunked file: {target_path.name}")
            continue
        
        raw_chunks = box_rag.chunk_markdown_file(md_file)
        enriched_chunks = box_rag.add_embeddings(chunks=raw_chunks, text_encoder=text_encoder)
        save_json_to_file(content=enriched_chunks, target_path=target_path)
        logger.info(f"chunks saved to file -> {target_path}")

        # Run database operation using chunk+embeddings file 
        db_config: DatabaseConfig = get_database_config()
        db_string = db_config.get_uri() 

        chunks = read_json_file(target_path)
        box_rag.store_doc_chunks(
            document_slug=doc_slug,
            document_name=doc_name, 
            document_metadata= {
                "source_file": md_file.name,
                "pipeline": "bge-m3-rag",
            },
            chunks=chunks,
            conn_string=db_string)

        logger.info(f"chunks saved to pg_vector")
        # give CPU a breather!
        time.sleep(4)


                
def main():
   
    print(" ##################################### ")
    print(" @@@  CHUNKS BATCH SCRIPT   @@@        ")
    print(" @@@  Press CTRL+C to abort        @@@ ")
    print(" ##################################### ")
    
    time.sleep(1)

    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    # @IMP: check SRC and TARGET_DIR
    SRC_DIR = DOC_DIR / "smp" / "pp"
    TARGET_DIR = XOUT_DIR / "chunks"
    
    # Local text encoder 
    # LOCAL_BGEM3_PATH = "/home/rjha/code/models/bge-m3"
    # text_encoder = LocalBGEM3Encoder(model_path=LOCAL_BGEM3_PATH)

    # api text encoder
    api_keys = AppConfig.get("api_keys")
    do_api_client = DigitalOceanClient(api_key=api_keys["DO_INFERENCE_KEY"])
    text_encoder = DOBGEM3Encoder(api_client=do_api_client)
    # run the batch 
    process_batch(source_dir=SRC_DIR, target_dir=TARGET_DIR, text_encoder=text_encoder)


if __name__ == "__main__":
    # Register the SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)
    main()
