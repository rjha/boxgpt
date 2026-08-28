import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any
import numpy as np


def get_vector_candidates(
    query_embedding: list[float],
    db_config: dict,
    limit: int = 20,
    doc_id: str = "mai_out"
) -> List[Dict[str, Any]]:
    """Retrieves top K candidates ordered by vector cosine distance."""
    conn = psycopg2.connect(**db_config)
    register_vector(conn)
    cursor = conn.cursor()

    sql_query = """
        SELECT 
            id, section_title, page_start, page_end, content, embedding,
            1 - (embedding <=> %s::vector) AS vector_similarity
        FROM document_chunks
        WHERE doc_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    cursor.execute(sql_query, (query_embedding, doc_id, query_embedding, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for rank, row in enumerate(rows, start=1):
        results.append({
            "id": row[0],
            "section_title": row[1],
            "page_start": row[2],
            "page_end": row[3],
            "content": row[4],
            "embedding": row[5],
            "vector_similarity": row[6],
            "vector_rank": rank  # 1-based rank
        })

    return results


def get_lexical_candidates(
    query_str: str,
    db_config: dict,
    limit: int = 20,
    doc_id: str = "mai_out"
) -> List[Dict[str, Any]]:
    """Retrieves top K candidates ordered by tsvector full-text match score."""
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    sql_query = """
        SELECT 
            id, section_title, page_start, page_end, content, embedding,
            ts_rank_cd(to_tsvector('hindi', content), plainto_tsquery('hindi', %s)) AS lexical_score
        FROM document_chunks
        WHERE doc_id = %s 
          AND to_tsvector('hindi', content) @@ plainto_tsquery('hindi', %s)
        ORDER BY lexical_score DESC
        LIMIT %s;
    """

    cursor.execute(sql_query, (query_str, doc_id, query_str, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for rank, row in enumerate(rows, start=1):
        results.append({
            "id": row[0],
            "section_title": row[1],
            "page_start": row[2],
            "page_end": row[3],
            "content": row[4],
            "embedding": row[5],
            "lexical_score": row[6],
            "lexical_rank": rank  # 1-based rank
        })

    return results


def compute_rrf_in_python(
    vector_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    k: int = 60,
    vector_weight: float = 1.0,
    lexical_weight: float = 1.0,
    top_candidates_limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Combines independent Vector and Lexical result sets using 
    Reciprocal Rank Fusion (RRF) in Python.
    """
    combined_records: Dict[int, Dict[str, Any]] = {}
    rrf_scores: Dict[int, float] = {}

    # 1. Process Vector Search Ranks
    for item in vector_results:
        item_id = item["id"]
        rank = item["vector_rank"]
        combined_records[item_id] = item
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (vector_weight / (k + rank))

    # 2. Process Lexical Search Ranks
    for item in lexical_results:
        item_id = item["id"]
        rank = item["lexical_rank"]
        
        # If not present from vector search, add metadata record
        if item_id not in combined_records:
            combined_records[item_id] = item
            
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (lexical_weight / (k + rank))

    # 3. Attach final RRF score to items
    for item_id, score in rrf_scores.items():
        combined_records[item_id]["rrf_score"] = score

    # 4. Sort candidates by RRF score descending
    sorted_candidates = sorted(
        combined_records.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return sorted_candidates[:top_candidates_limit]


def execute_hybrid_search(
    query_str: str,
    query_embedding: list[float],
    db_config: dict,
    k: int = 60,
    candidate_limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Orchestrates fetching vector + keyword candidates from DB 
    and running RRF fusion in Python.
    """
    # 1. Fetch top candidates from both search mechanisms in DB
    vector_candidates = get_vector_candidates(query_embedding, db_config, limit=candidate_limit)
    lexical_candidates = get_lexical_candidates(query_str, db_config, limit=candidate_limit)

    # 2. Perform RRF fusion in Python
    merged_candidates = compute_rrf_in_python(
        vector_results=vector_candidates,
        lexical_results=lexical_candidates,
        k=k,
        top_candidates_limit=candidate_limit
    )

    return merged_candidates


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def apply_mmr_filter(
    candidates: List[Dict[str, Any]],
    query_embedding: list[float],
    final_top_k: int = 4,
    lambda_param: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Applies Maximal Marginal Relevance (MMR) to balance query relevance 
    and result diversity among candidate vectors.
    """
    if not candidates:
        return []

    q_vec = np.array(query_embedding, dtype=np.float32)
    cand_vecs = [np.array(c["embedding"], dtype=np.float32) for c in candidates]

    # Precalculate similarity to query for all candidates
    query_sims = [cosine_similarity(q_vec, c_vec) for c_vec in cand_vecs]

    selected_indices = []
    unselected_indices = list(range(len(candidates)))

    while unselected_indices and len(selected_indices) < final_top_k:
        best_score = -float('inf')
        best_idx = -1

        for idx in unselected_indices:
            sim_to_query = query_sims[idx]
            
            # Find maximum similarity to any already selected candidate
            if not selected_indices:
                max_sim_to_selected = 0.0
            else:
                max_sim_to_selected = max(
                    cosine_similarity(cand_vecs[idx], cand_vecs[s_idx])
                    for s_idx in selected_indices
                )

            # MMR Equation formula
            mmr_score = (lambda_param * sim_to_query) - ((1 - lambda_param) * max_sim_to_selected)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]
