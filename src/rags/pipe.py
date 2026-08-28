


def run_retrieval_pipeline(user_query: str) -> str:
    # 1. Query Processing
    query_emb = process_user_query(user_query)

    # 2. Database Load
    db_config = load_db_config()

    # 3. Hybrid Search with RRF (Top 20 candidates)
    candidates = execute_hybrid_search_rrf(
        query_str=user_query,
        query_embedding=query_emb,
        db_config=db_config,
        k=60,
        candidate_limit=20
    )

    # 4. Diversity Filter via MMR (Select 4 diverse chunks)
    diverse_chunks = apply_mmr_filter(
        candidates=candidates,
        query_embedding=query_emb,
        final_top_k=4,
        lambda_param=0.7
    )

    # 5. Format Prompt for LLM
    final_prompt = format_llm_prompt(user_query, diverse_chunks)
    return final_prompt



