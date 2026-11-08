import logging
from pathlib import Path

import softmaxx.boxgpt.rag as box_rag 
from softmaxx.boxgpt.client.llm import DigitalOceanClient 

from softmaxx.config import (
    AppConfig, 
    get_logger_config, 
    get_database_config,
    DatabaseConfig
)

from softmaxx.boxgpt.encoder import (
    LocalBGEM3Encoder, 
    DOBGEM3Encoder
)

from scripts.chunk.common import (
    read_json_file,
    save_json_to_file,
    transform_slug
)


def main():
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)
    
    # @IMP: verify source, target and model paths
    BUILD_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()
    BACKUP_DIR = Path('/home/rjha/backup/ab-out/smp/pp')
    FILE_NAME = "smpe_i018_maintenance_procedure_for_lt_breakers.md"
    SRC_PATH = BACKUP_DIR / FILE_NAME
    TARGET_PATH = BUILD_DIR / "chunk" / "ab_pp_chunk.json"
    
    # Local text encoder 
    # LOCAL_BGEM3_PATH = "/home/rjha/code/models/bge-m3"
    # text_encoder = LocalBGEM3Encoder(model_path=LOCAL_BGEM3_PATH)

    # api text encoder
    api_keys = AppConfig.get("api_keys")
    do_api_client = DigitalOceanClient(api_key=api_keys["DO_INFERENCE_KEY"])
    text_encoder = DOBGEM3Encoder(api_client=do_api_client)

    # step by step verification
    # STEP 1. read and chunk markdown file
    # STEP 2. add embeddings
    # STEP 3. save the embeddings into file
    raw_chunks = box_rag.chunk_markdown_file(SRC_PATH)
    enriched_chunks = box_rag.add_embeddings(chunks=raw_chunks, text_encoder=text_encoder)
    save_json_to_file(content=enriched_chunks, target_path=TARGET_PATH)
    
    db_config:DatabaseConfig = get_database_config()
    db_string = db_config.get_uri() 
    doc_slug = SRC_PATH.stem 
    doc_name = transform_slug(doc_slug)
    chunks = read_json_file(TARGET_PATH)

    box_rag.store_doc_chunks(
            document_slug=doc_slug,
            document_name=doc_name, 
            document_metadata= {
                "source_file": SRC_PATH.name,
                "pipeline": "bge-m3-rag",
            },
            chunks=chunks,
            conn_string=db_string)
    
    
if __name__ == "__main__":
    main()