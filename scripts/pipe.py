from softmaxx.rag.query import format_llm_prompt, process_user_query
from softmaxx.rag.search import execute_hybrid_search, apply_mmr_filter
from softmaxx.config import AppConfig, get_logger_config


def run_pipeline(user_query: str, query_config:dict[str, any]) -> str:
    # 1. Query Processing
    query_emb = process_user_query(user_query, query_config=query_config)
    print("query embedding geneated")

    # 3. Hybrid Search with RRF (Top 20 candidates)
    candidates = execute_hybrid_search(
        query_str=user_query, 
        query_config=query_config,
        query_embedding=query_emb,
        k=60,
        candidate_limit=20
    )

    print("Hybrid search done!")
    # 4. Diversity Filter via MMR (Select 4 diverse chunks)
    diverse_chunks = apply_mmr_filter(
        candidates=candidates,
        query_embedding=query_emb,
        final_top_k=4,
        lambda_param=0.7
    )

    print("MMR filter applied!")
    # 5. Format Prompt for LLM
    final_prompt = format_llm_prompt(user_query, diverse_chunks)
    return final_prompt

def main():
    query_config = {
        "doc_id": "ICAR_MAG_JAN2026",
        "model_path": "/home/rjha/code/models/bge-m3"
    }

    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)
    user_query = "छत पर खेती कैसे कर सकते हैं?"
    final_prompt = run_pipeline(user_query, query_config)
    print("---------- LLM prompt ---------")
    print(final_prompt)

 
if __name__ == "__main__":
    main()
    