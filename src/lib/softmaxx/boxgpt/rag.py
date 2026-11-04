import json
import psycopg2
import logging
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
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

# Create Vector embeddings

def chunk_content(
    doc_path: Path, 
    max_chunk_chars: int = 1500, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Reads structured section JSON and produces a list of chunk dictionaries
    with section context prepended to the embedding text.
    """

    logger = logging.getLogger("main." + __name__)
    with doc_path.open("r", encoding="utf-8") as f:
        sections = json.load(f)

    # Recursive splitter for sections exceeding max_chunk_chars
    # @todo pass extra stop signs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks: List[Dict[str, Any]] = []

    for section in sections:
        section_title = section.get("section_title", "General")
        page_range = section.get("page_range", [1, 1])
        raw_content = section.get("content", "").strip()

        if not raw_content:
            continue

        # If section is small enough, keep as a single chunk
        if len(raw_content) <= max_chunk_chars:
            section_chunks = [raw_content]
        else:
            # Recursively split large sections into sub-chunks
            section_chunks = text_splitter.split_text(raw_content)

        for idx, chunk_text in enumerate(section_chunks):
            # Context Prepending: Crucial for vector search quality
            embedding_text = f"[भाग {section_title}]\n\n{chunk_text}"

            chunks.append({
                "section_title": section_title,
                "page_start": page_range[0],
                "page_end": page_range[1],
                "chunk_index": idx,
                "content": chunk_text,                # Original text (for UI display)
                "embedding_text": embedding_text,     # Contextualized text (for BGE-M3)
                "embedding": []                       # To be populated by get_chunk_embeddings
            })

    return chunks


def get_embeddings(
    chunks: List[Dict[str, Any]], 
    model_path: str = "./models/bge-m3", 
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Encodes chunk texts in batches using BAAI/bge-m3 and attaches 
    1024-dimensional float arrays to each chunk object.
    """
    if not chunks:
        return []

    logger = logging.getLogger("main." + __name__)
    logger.info(f"Loading transformer model from: {model_path}...")
    model = SentenceTransformer(model_path)

    # Extract all prepended texts into a flat list for batching
    texts_to_encode = [chunk["embedding_text"] for chunk in chunks]

    logger.info(f"Generating embeddings for {len(texts_to_encode)} chunks (Batch size: {batch_size})...")
    # Generate embeddings in a single batch pass (returns NumPy array)
    embeddings = model.encode(
        texts_to_encode, 
        batch_size=batch_size, 
        show_progress_bar=True,
        normalize_embeddings=True  # Normalizing simplifies cosine distance calculations
    )

    # Convert NumPy arrays to Python float lists and attach back to payload dicts
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks



def store_in_database(
    chunks: List[Dict[str, Any]], 
    doc_id: str
) -> None:
    """
    Batch inserts enriched chunks (metadata, text, and bge-m3 embeddings) 
    into PostgreSQL with pgvector.
    """

    logger = logging.getLogger("main." + __name__)

    if not chunks:
        logger.info("No chunks provided to store.")
        return

    logger.info(f"Connecting to PostgreSQL database...")
    db_config:DatabaseConfig = get_database_config()
    logger.info(f"Connecting to PostgreSQL {db_config}...")
    conn = psycopg2.connect(**db_config.get_map())
    
    try:
        # Register pgvector extension handler for psycopg2
        register_vector(conn)
        cursor = conn.cursor()

        # Prepare tuple records matching table schema
        insert_records = []
        for chunk in chunks:
            insert_records.append((
                doc_id,
                chunk["section_title"],
                chunk["page_start"],
                chunk["page_end"],
                chunk["chunk_index"],
                chunk["content"],
                chunk["embedding_text"],
                chunk["embedding"]  # pgvector converts Python list[float] to vector automatically
            ))

        sql_query = """
            INSERT INTO document_chunks (
                doc_id,
                section_title,
                page_start,
                page_end,
                chunk_index,
                content,
                embedding_text,
                embedding
            ) VALUES %s;
        """

        logger.info(f"Batch inserting {len(insert_records)} chunks into PostgreSQL...")
        
        # Execute batch insertion efficiently
        execute_values(cursor, sql_query, insert_records)
        
        # Commit transaction
        conn.commit()
        logger.info(f"Successfully stored {len(insert_records)} chunks in database!")

    except Exception as e:
        conn.rollback()
        logger.info(f"Error inserting chunks into database: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()


def process_user_query(raw_query: str, query_config:dict[str, any]) -> list[float]:
    """Encodes user query into vector space without modifying query text."""
    model_path = query_config["model_path"]
    model = SentenceTransformer(model_path)
    # Output is a normalized vector
    query_embedding = model.encode(raw_query, normalize_embeddings=True).tolist()
    return query_embedding


# search functions 

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

