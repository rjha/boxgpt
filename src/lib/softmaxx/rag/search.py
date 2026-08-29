import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any
from softmaxx.config import get_database_config, DatabaseConfig


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _normalize_embedding(raw_embedding: Any) -> list[float]:
    """
    Private helper to convert pgvector/ndarray/string representations
    from PostgreSQL into a standardized Python list[float].
    """
    if hasattr(raw_embedding, "tolist"):
        return [float(x) for x in raw_embedding.tolist()]
    elif hasattr(raw_embedding, "to_numpy"):
        return [float(x) for x in raw_embedding.to_numpy().tolist()]
    elif isinstance(raw_embedding, str):
        return [float(x) for x in raw_embedding.strip("[]").split(",")]
    return [float(x) for x in list(raw_embedding)]


def _get_vector_candidates(
    query_embedding: list[float],
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves top K candidates ordered by vector cosine distance."""

    db_config:DatabaseConfig = get_database_config()
    conn = psycopg2.connect(**db_config.get_map())
    register_vector(conn)
    cursor = conn.cursor()

    sql_query = """
        SELECT 
            id, section_title, page_start, page_end, content, embedding,
            1 - (embedding <=> %s::vector) AS vector_similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    cursor.execute(sql_query, (query_embedding, query_embedding, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for rank, row in enumerate(rows, start=1):
        clean_embedding = _normalize_embedding(row[5])
        results.append({
            "id": row[0],
            "section_title": row[1],
            "page_start": row[2],
            "page_end": row[3],
            "content": row[4],
            "embedding": clean_embedding,
            "vector_similarity": float(row[6]),
            "vector_rank": rank  # 1-based rank
        })

    return results


def _get_lexical_candidates(
    query_str: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves top K candidates ordered by tsvector full-text match score."""
    
    db_config:DatabaseConfig = get_database_config()
    conn = psycopg2.connect(**db_config.get_map())

    cursor = conn.cursor()

    sql_query = """
        SELECT 
            id, section_title, page_start, page_end, content, embedding,
            ts_rank_cd(to_tsvector('hindi', content), plainto_tsquery('hindi', %s)) AS lexical_score
        FROM document_chunks
        WHERE  to_tsvector('hindi', content) @@ plainto_tsquery('hindi', %s)
        ORDER BY lexical_score DESC
        LIMIT %s;
    """

    cursor.execute(sql_query, (query_str, query_str, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for rank, row in enumerate(rows, start=1):
        clean_embedding = _normalize_embedding(row[5])
        results.append({
            "id": row[0],
            "section_title": row[1],
            "page_start": row[2],
            "page_end": row[3],
            "content": row[4],
            "embedding": clean_embedding,
            "lexical_score": float(row[6]),
            "lexical_rank": rank  # 1-based rank
        })

    return results


def _compute_rrf_in_python(
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
    combined_records: Dict[Any, Dict[str, Any]] = {}
    rrf_scores: Dict[Any, float] = {}

    # 1. Process Vector Search Ranks
    for item in vector_results:
        item_id = item["id"]
        rank = item["vector_rank"]
        # Make a shallow copy to prevent mutating raw input dicts
        combined_records[item_id] = dict(item)
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (vector_weight / (k + rank))

    # 2. Process Lexical Search Ranks
    for item in lexical_results:
        item_id = item["id"]
        rank = item["lexical_rank"]
        
        if item_id not in combined_records:
            combined_records[item_id] = dict(item)
        else:
            # Merge lexical metadata into existing vector record
            combined_records[item_id]["lexical_rank"] = rank
            if "lexical_score" in item:
                combined_records[item_id]["lexical_score"] = item["lexical_score"]
            
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




# #####################################
# PUBLIC METHODS 
#
#######################################

def execute_hybrid_search(
    query_str: str,
    query_config:dict[str, any],
    query_embedding: list[float],
    k: int = 60,
    candidate_limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Orchestrates fetching vector + keyword candidates from DB 
    and running RRF fusion in Python.
    """
    # 1. Fetch top candidates from both search mechanisms in DB
    vector_candidates = _get_vector_candidates(query_embedding, limit=candidate_limit)
    lexical_candidates = _get_lexical_candidates(query_str, limit=candidate_limit)

    # 2. Perform RRF fusion in Python
    merged_candidates = _compute_rrf_in_python(
        vector_results=vector_candidates,
        lexical_results=lexical_candidates,
        k=k,
        top_candidates_limit=candidate_limit
    )

    return merged_candidates


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

    # Safe direct conversion since DAL guarantees list[float]
    q_vec = np.array(query_embedding, dtype=np.float32)
    cand_vecs = [np.array(c["embedding"], dtype=np.float32) for c in candidates]

    # Precalculate similarity to query for all candidates
    query_sims = [_cosine_similarity(q_vec, c_vec) for c_vec in cand_vecs]

    selected_indices: List[int] = []
    unselected_indices = list(range(len(candidates)))

    # Cache for max similarity between unselected candidate and already selected set
    # Maps candidate index -> max similarity to any selected candidate
    max_sim_to_selected = {idx: 0.0 for idx in unselected_indices}

    while unselected_indices and len(selected_indices) < final_top_k:
        best_score = -float('inf')
        best_idx = -1

        for idx in unselected_indices:
            sim_to_query = query_sims[idx]
            sim_to_set = max_sim_to_selected[idx] if selected_indices else 0.0

            # MMR Equation: λ * Sim(Query, Doc) - (1 - λ) * MaxSim(Doc, SelectedSet)
            mmr_score = (lambda_param * sim_to_query) - ((1 - lambda_param) * sim_to_set)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        # Update tracking lists
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

        # Update cached max similarity to selected set for remaining candidates
        newly_selected_vec = cand_vecs[best_idx]
        for idx in unselected_indices:
            sim_to_new = _cosine_similarity(cand_vecs[idx], newly_selected_vec)
            if sim_to_new > max_sim_to_selected[idx]:
                max_sim_to_selected[idx] = sim_to_new

    return [candidates[i] for i in selected_indices]
