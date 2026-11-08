import logging

# softmaxx libraries
import softmaxx.config as s_config
import softmaxx.boxgpt.rag as box_rag
import softmaxx.boxgpt.prompt as box_prompt
from softmaxx.boxgpt.client.llm import DigitalOceanClient
from softmaxx.boxgpt.encoder import TextEncoder, LocalBGEM3Encoder


def main():
    s_config.AppConfig.load()
    log_config = s_config.get_logger_config("local")
    s_config.AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)

    text_encoder:TextEncoder = LocalBGEM3Encoder(model_path="/home/rjha/code/models/bge-m3")

    # @IMP: user query 
    user_query = "what are the PPE requirements for compressor maintenance?"
    db_config:s_config.DatabaseConfig = s_config.get_database_config()
    db_string = db_config.get_uri() 
    mmr_chunks = box_rag.search_doc_chunks(
        db_string=db_string, 
        text_encoder=text_encoder,
        user_query=user_query
    )

    llm_messages = box_prompt.rag_query_message(user_query, mmr_chunks)
    api_keys = s_config.AppConfig.get("api_keys")

    llm_client = DigitalOceanClient(api_key=api_keys["DO_INFERENCE_KEY"])
    answer = llm_client.sonnet_55_chat(model_name="anthropic-claude-sonnet-5.5", llm_messages=llm_messages)
    print(answer)

    
if __name__ == "__main__":
    main()