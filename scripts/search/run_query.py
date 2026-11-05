import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import softmaxx.config as s_config
import softmaxx.boxgpt.rag as box_rag


def main():
    s_config.AppConfig.load()
    log_config = s_config.get_logger_config("local")
    s_config.AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)

    MODEL_PATH = "/home/rjha/code/models/bge-m3"
    model_instance = SentenceTransformer(MODEL_PATH)
    print(f" loaded model from {MODEL_PATH}")
    
    user_query = "How can I fix a Boiler tube?"
    query_embedding = model_instance.encode(user_query, normalize_embeddings=True).tolist()
    print("query embedding geneated")

    #  Hybrid Search with RRF (Top 20 candidates)
    db_config:s_config.DatabaseConfig = s_config.get_database_config()
    db_string = db_config.get_uri() 

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

    # format LLM prompt using user_query and mmr_chunks
    print("MMR filter applied!")

    # 5. Format LLM prompt using externalized files and MMR chunks
    PROMPT_DIR = Path("prompts")
    
    # Read system prompt and user template from files
    system_prompt = (PROMPT_DIR / "system_rag.txt").read_text(encoding="utf-8").strip()
    user_template = (PROMPT_DIR / "user_rag_template.txt").read_text(encoding="utf-8")

    # Build structured context string from MMR chunks
    context_builder = []
    for i, chunk in enumerate(mmr_chunks, start=1):
        source = chunk.get("source_name", "Unknown Document")
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

    # Final payload ready to pass to your LLM client (e.g., OpenAI, Ollama, etc.)
    logger.info("LLM prompts successfully formatted.")
    print("\n--- SYSTEM PROMPT ---")
    print(system_prompt)
    print("\n--- FORMATTED USER PROMPT ---")
    print(formatted_user_prompt)
    
if __name__ == "__main__":
    main()