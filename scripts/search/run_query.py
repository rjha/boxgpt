import logging
from sentence_transformers import SentenceTransformer

# softmaxx libraries
import softmaxx.config as s_config
import softmaxx.boxgpt.rag as box_rag
import softmaxx.boxgpt.prompt as box_prompt
from softmaxx.boxgpt.client.llm import DigitalOceanClient


def main():
    s_config.AppConfig.load()
    log_config = s_config.get_logger_config("local")
    s_config.AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)

    MODEL_PATH = "/home/rjha/code/models/bge-m3"
    model_instance = SentenceTransformer(MODEL_PATH)
    print(f" loaded model from {MODEL_PATH}")

    # @IMP: user query 
    user_query = "what are the pre-checks for compressor maintenance?"
    db_config:s_config.DatabaseConfig = s_config.get_database_config()
    db_string = db_config.get_uri() 
    mmr_chunks = box_rag.search_doc_chunks(
        db_string=db_string, 
        model=model_instance, 
        user_query=user_query
    )

    llm_messages = box_prompt.rag_query_message(user_query, mmr_chunks)
    llm_client = DigitalOceanClient("DO_SONNET_055")
    answer = llm_client.send_message(llm_messages=llm_messages)
    print(answer)

    
if __name__ == "__main__":
    main()