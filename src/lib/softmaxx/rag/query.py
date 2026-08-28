
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from softmaxx.rag.search import execute_hybrid_search, apply_mmr_filter



def process_user_query(raw_query: str, model_path: str = "./models/bge-m3") -> list[float]:
    """Encodes user query into vector space without modifying query text."""
    model = SentenceTransformer(model_path)
    # Output is a normalized vector
    query_embedding = model.encode(raw_query, normalize_embeddings=True).tolist()
    return query_embedding


def format_llm_prompt(raw_query: str, selected_chunks: List[Dict[str, Any]]) -> str:
    """Formats system prompt, retrieved diverse contexts, and user query."""
    context_blocks = []
    for chunk in selected_chunks:
        block = (
            f"[दस्तावेज़ भाग: {chunk['section_title']} | पृष्ठ: {chunk['page_start']}-{chunk['page_end']}]\n"
            f"{chunk['content']}"
        )
        context_blocks.append(block)

    joined_context = "\n\n---\n\n".join(context_blocks)

    system_instruction = (
        "आप एक कुशल बोनसाई सहायक हैं। केवल नीचे दिए गए संदर्भ (Context) के आधार पर उत्तर दें।\n"
        "यदि उत्तर संदर्भ में उपलब्ध नहीं है, तो स्पष्ट कहें: 'दिए गए दस्तावेज़ में इसकी जानकारी उपलब्ध नहीं है।'\n"
        "अपनी ओर से कोई भी झूठी या मनगढ़ंत जानकारी न जोड़ें। उत्तर स्पष्ट हिंदी में दें।"
    )

    prompt_payload = f"""System: {system_instruction}

    Context Excerpts:
    ---
    {joined_context}
    ---

    User Query: {raw_query}
    """
    return prompt_payload




def run_retrieval_pipeline(user_query: str) -> str:
    # 1. Query Processing
    query_emb = process_user_query(user_query)

    
    # 3. Hybrid Search with RRF (Top 20 candidates)
    candidates = execute_hybrid_search(
        query_str=user_query,
        query_embedding=query_emb,
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

