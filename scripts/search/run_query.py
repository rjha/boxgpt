import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import softmaxx.config as s_config
import softmaxx.boxgpt.rag as box_rag


def get_mmr_chunks(db_string, model, user_query):
    query_embedding = model.encode(user_query, normalize_embeddings=True).tolist()
    print("query embedding geneated")
    #  Hybrid Search with RRF (Top 20 candidates)
    candidates = box_rag.execute_hybrid_search(
        query_str=user_query,
        query_embedding=query_embedding,
        conn_string=db_string,
        candidate_limit=20
    )

    print("Hybrid search done!")
    # 4. Diversity Filter via MMR (Select 4 diverse chunks)
    mmr_chunks = box_rag.apply_mmr_filter(
        candidates=candidates,
        query_embedding=query_embedding,
        final_top_k=4,
        lambda_param=0.7
    )
    
    print("MMR filter applied!")
    return mmr_chunks 


def get_llm_message(
    system_prompt, 
    user_template, 
    user_query, 
    chunks
):

    # Build structured context string from MMR chunks
    context_builder = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("doc_name", "No Document")
        section = chunk.get("section_title", "General")
        content = chunk.get("content", "").strip()
        
        context_builder.append(
            f"[Snippet {i} | Source: {source} | Section: {section}]\n{content}\n"
        )
    
    formatted_context = "\n".join(context_builder)

    # Format the user prompt template with context and query
    formatted_user_prompt = user_template.format(
        context=formatted_context,
        user_query=user_query
    )

    return  [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_user_prompt}
    ]



def main():
    s_config.AppConfig.load()
    log_config = s_config.get_logger_config("local")
    s_config.AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)

    MODEL_PATH = "/home/rjha/code/models/bge-m3"
    model_instance = SentenceTransformer(MODEL_PATH)
    print(f" loaded model from {MODEL_PATH}")

    # @IMP: user query 
    user_query = "How can I fix a Boiler tube?"
    db_config:s_config.DatabaseConfig = s_config.get_database_config()
    db_string = db_config.get_uri() 
    mmr_chunks = get_mmr_chunks(
        db_string=db_string, 
        model=model_instance, 
        user_query=user_query
    )

    # 5. Format LLM prompt using prompts and MMR chunks
    PROMPT_DIR = Path("~/code/github/boxgpt/scripts/search/prompts").expanduser()
    # Read system prompt and user template from files
    system_prompt = (PROMPT_DIR / "system_rag.txt").read_text(encoding="utf-8").strip()
    user_template = (PROMPT_DIR / "user_rag_template.txt").read_text(encoding="utf-8")

    
    llm_messages = get_llm_message(
        system_prompt=system_prompt, 
        user_template=user_template, 
        user_query=user_query,
        chunks=mmr_chunks)

    print(llm_messages)


    
if __name__ == "__main__":
    main()