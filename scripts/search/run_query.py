import logging

# softmaxx libraries
from softmaxx.config import (
    AppConfig, 
    get_logger_config, 
    DatabaseConfig, 
    get_database_config
)

import softmaxx.boxgpt.rag as box_rag
import softmaxx.boxgpt.prompt as box_prompt
from softmaxx.boxgpt.client.llm import DigitalOceanClient
from softmaxx.boxgpt.encoder import (
    TextEncoder, 
    LocalBGEM3Encoder, 
    DOBGEM3Encoder
)

from scripts.constants import LOCAL_BGEM3_PATH


def main():
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)

  
    # Local text encoder 
    # LOCAL_BGEM3_PATH = "/home/rjha/code/models/bge-m3"
    # text_encoder = LocalBGEM3Encoder(model_path=LOCAL_BGEM3_PATH)

    # api text encoder
    api_keys = AppConfig.get("api_keys")
    do_api_client = DigitalOceanClient(api_key=api_keys["DO_INFERENCE_KEY"])
    text_encoder = DOBGEM3Encoder(api_client=do_api_client)

    
    # @IMP: user query 
    user_query = "what are the PPE requirements for compressor maintenance?"
    db_config: DatabaseConfig = get_database_config()
    db_string = db_config.get_uri() 
    mmr_chunks = box_rag.search_doc_chunks(
        db_string=db_string, 
        text_encoder=text_encoder,
        user_query=user_query
    )

    llm_messages = box_prompt.rag_query_message(user_query, mmr_chunks)
    api_keys = AppConfig.get("api_keys")

    llm_client = DigitalOceanClient(api_key=api_keys["DO_INFERENCE_KEY"])
    answer = llm_client.sonnet_55_chat(llm_messages=llm_messages)
    print(answer)

    
if __name__ == "__main__":
    main()