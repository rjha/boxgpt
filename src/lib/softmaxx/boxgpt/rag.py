import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
# postgres 
import psycopg
from pgvector.psycopg import register_vector
# embeddings 
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter



@dataclass(frozen=True)
class RRFTuningConfig:
    k: int = 60
    vector_weight: float = 1.0
    lexical_weight: float = 1.0
    top_candidates_limit: int = 20



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
    conn_string: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves top K candidates ordered by vector cosine distance."""

    sql_query = """
    SELECT 
        c.id,
        c.doc_id,
        c.section_title,
        c.page_start,
        c.page_end,
        c.chunk_index,
        s.doc_name,
        s.doc_slug,
        s.doc_meta,
        c.content,
        c.embedding,
        (1 - (c.embedding <=> %s::vector)) AS vector_score
    FROM doc_chunks c
    JOIN doc_source s ON c.doc_id = s.id
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s;       
    """

    with psycopg.connect(conn_string) as conn:
        register_vector(conn)
        with conn.cursor() as cursor:
            cursor.execute(sql_query, (query_embedding, query_embedding, limit))
            rows = cursor.fetchall()

    # starting with rank=0 will blow up RRF function 
    # Hence we assign rank=1 to the first candiate in list 
    # RRF formula has 1 / (k + rank)
    candidates = []
    for rank, row in enumerate(rows, start=1):
        raw_embedding = row[10]
        vector_similarity = float(row[11])
        clean_embedding = _normalize_embedding(raw_embedding)

        candidates.append({
            "id": row[0],
            "doc_id" : row[1],
            "section_title": row[2],
            "page_start": row[3],
            "page_end": row[4],
            "chunk_index": row[5],
            "doc_name": row[6],
            "doc_slug": row[7],
            "doc_meta": row[8],
            "content": row[9],
            "embedding": clean_embedding,
            "vector_similarity": vector_similarity,
            "vector_rank": rank
        })

    return candidates


def _get_lexical_candidates(
    query_str: str,
    conn_string: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves top K candidates ordered by tsvector full-text match score."""

    sql_query = """
    SELECT 
        c.id,
        c.doc_id,
        c.section_title,
        c.page_start,
        c.page_end,
        c.chunk_index,
        s.doc_name,
        s.doc_slug,
        s.doc_meta,
        c.content,
        c.embedding,
        ts_rank_cd(to_tsvector('english', c.content), websearch_to_tsquery('english', %s)) AS lexical_score
    FROM doc_chunks c
    JOIN doc_source s ON c.doc_id = s.id
    WHERE to_tsvector('english', c.content) @@ websearch_to_tsquery('english', %s)
    ORDER BY lexical_score DESC
    LIMIT %s;
    """
    
    with psycopg.connect(conn_string) as conn:
        register_vector(conn)
        with conn.cursor() as cursor:
            cursor.execute(sql_query, (query_str, query_str, limit))
            rows = cursor.fetchall()

    candidates = []
    for rank, row in enumerate(rows, start=1):
        raw_embedding = row[10]
        clean_embedding = _normalize_embedding(raw_embedding)
        lexical_score = float(row[11])
        
        candidates.append({
            "id": row[0],
            "doc_id" : row[1],
            "section_title": row[2],
            "page_start": row[3],
            "page_end": row[4],
            "chunk_index": row[5],
            "doc_name": row[6],
            "doc_slug": row[7],
            "doc_meta": row[8],
            "content": row[9],
            "embedding": clean_embedding,
            "lexical_score": lexical_score,
            "lexical_rank": rank   
        })

    return candidates


def _compute_rrf(
    vector_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    top_candidates_limit: int = 20,
    config: RRFTuningConfig = RRFTuningConfig()
) -> List[Dict[str, Any]]:
    """
    Combines independent Vector and Lexical result sets using 
    Reciprocal Rank Fusion (RRF)
    """
    combined_records: Dict[Any, Dict[str, Any]] = {}
    rrf_scores: Dict[Any, float] = {}

    # Process Vector Search Ranks
    for item in vector_results:
        item_id = item["id"]
        rank = item["vector_rank"]
        # Make a shallow copy to prevent mutating input dicts
        combined_records[item_id] = dict(item)
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (config.vector_weight / (config.k + rank))

    # Process Lexical Search Ranks
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
            
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (config.lexical_weight / (config.k + rank))

    # Attach final RRF score to items
    for item_id, score in rrf_scores.items():
        combined_records[item_id]["rrf_score"] = score

    # Sort candidates by RRF score descending
    sorted_candidates = sorted(
        combined_records.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return sorted_candidates[:config.top_candidates_limit]


# @todo remove - use boxgpt.parse module 
def _parse_markdown_to_sections(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Parses a cleaned markdown document into sections based on headers (#, ##, ###)
    and tracks approximate page numbers from '--- Page X ---' markers.
    """
    sections = []
    current_title = "Introduction"
    current_page = 1
    current_content_lines = []

    lines = markdown_text.splitlines()
    for line in lines:
        # Track page numbers if present in markdown
        page_match = re.match(r"^---\s+Page\s+(\d+)\s+---", line.strip(), re.IGNORECASE)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Detect markdown headers
        header_match = re.match(r"^(#{1,4})\s+(.+)", line.strip())
        if header_match:
            # If we already accumulated content for the previous section, save it
            if current_content_lines:
                sections.append({
                    "section_title": current_title,
                    "page_range": [current_page, current_page],
                    "content": "\n".join(current_content_lines).strip()
                })
                current_content_lines = []
            
            current_title = header_match.group(2).strip()
            continue

        current_content_lines.append(line)

    # Append the final section
    if current_content_lines:
        sections.append({
            "section_title": current_title,
            "page_range": [current_page, current_page],
            "content": "\n".join(current_content_lines).strip()
        })

    return sections


def _execute_hybrid_search(
    query_str: str,
    query_embedding: list[float],
    conn_string: str,
    candidate_limit: int = 20,
    rrf_config: RRFTuningConfig = RRFTuningConfig()
) -> List[Dict[str, Any]]:
    """
    Orchestrates fetching vector + keyword candidates from DB 
    and running RRF fusion in Python.
    """
    # 1. Fetch top candidates from both search mechanisms in DB passing conn_string
    vector_candidates = _get_vector_candidates(query_embedding, conn_string=conn_string, limit=candidate_limit)
    lexical_candidates = _get_lexical_candidates(query_str, conn_string=conn_string, limit=candidate_limit)

    # 2. Perform RRF fusion using RRF fusion config
    merged_candidates = _compute_rrf(
        vector_results=vector_candidates,
        lexical_results=lexical_candidates,
        top_candidates_limit=candidate_limit,
        config=rrf_config
    )

    return merged_candidates



def _apply_mmr_filter(
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


# #####################################
# PUBLIC METHODS 
#
#######################################

# Create Vector embeddings
def chunk_markdown_content(
    doc_path: Path, 
    max_chunk_chars: int = 1500, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Reads a cleaned markdown file, parses it into sections, and produces 
    a list of chunk dictionaries with English section context prepended.
    """
    markdown_text = doc_path.read_text(encoding="utf-8")
    sections = _parse_markdown_to_sections(markdown_text)

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

        if len(raw_content) <= max_chunk_chars:
            section_chunks = [raw_content]
        else:
            section_chunks = text_splitter.split_text(raw_content)

        for idx, chunk_text in enumerate(section_chunks):
            # Updated to English prefix for BGE-M3 compatibility
            embedding_text = f"[Section: {section_title}]\n\n{chunk_text}"

            chunks.append({
                "section_title": section_title,
                "page_start": page_range[0],
                "page_end": page_range[1],
                "chunk_index": idx,
                "content": chunk_text,
                "embedding_text": embedding_text,
                "embedding": []  # To be populated in Step 2
            })

    return chunks


def add_embeddings(
    chunks: List[Dict[str, Any]], 
    model: SentenceTransformer, 
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Encodes chunk texts in batches using BAAI/bge-m3 and attaches 
    1024-dimensional float arrays to each chunk object.
    """
    if not chunks:
        return []

    logger = logging.getLogger("main." + __name__)
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



def store_doc_chunks(
    document_slug: str,
    document_name: str, 
    document_metadata: Dict[Any, Any], 
    chunks: List[Dict[str, Any]], 
    conn_string: str
):
    """
    Inserts a source document and its associated chunks with BGE-M3 vectors 
    into a PostgreSQL database using pgvector.
    """
    
    logger = logging.getLogger("main." + __name__)
    with psycopg.connect(conn_string) as conn:
        register_vector(conn)  # vector adapter for psycopg v3
        with conn.cursor() as cur:
            # Insert or get source document ID
            cur.execute(
                """
                INSERT INTO doc_source (doc_slug, doc_name, doc_meta)
                VALUES (%s, %s, %s)
                ON CONFLICT (doc_slug) 
                DO UPDATE SET doc_meta = EXCLUDED.doc_meta
                RETURNING id;
                """,
                (document_slug, document_name, psycopg.types.json.Json(document_metadata))
            )

            doc_id = cur.fetchone()[0]
            # Clear existing chunks for this document if re-processing
            cur.execute("DELETE FROM doc_chunks WHERE doc_id = %s;", (doc_id,))

            # Prepare batch insert data for chunks
            chunk_records = []
            for chunk in chunks:
                chunk_records.append((
                    doc_id,
                    chunk.get("section_title"),
                    chunk.get("page_start"),
                    chunk.get("page_end"),
                    chunk.get("chunk_index"),
                    chunk.get("content"),
                    chunk.get("embedding_text"),
                    chunk.get("embedding")  # psycopg will do lists => VECTOR
                ))

            # Perform batch insertion
            cur.executemany(
                """
                INSERT INTO doc_chunks (
                    doc_id, section_title, page_start, page_end, 
                    chunk_index, content, embedding_text, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """,
                chunk_records
            )
            conn.commit()
            logger.info(f"stored {len(chunk_records)} chunks for '{document_name}' in pg_vector.")



def search_doc_chunks(
        db_string:str, 
        model:SentenceTransformer, 
        user_query:str
    ):

    logger = logging.getLogger("main." + __name__)
    query_embedding = model.encode(user_query, normalize_embeddings=True).tolist()
    
    #  Hybrid Search with RRF (Top 20 candidates)
    candidates = _execute_hybrid_search(
        query_str=user_query,
        query_embedding=query_embedding,
        conn_string=db_string,
        candidate_limit=20
    )


    # 4. Diversity Filter via MMR (Select 4 diverse chunks)
    mmr_chunks = _apply_mmr_filter(
        candidates=candidates,
        query_embedding=query_embedding,
        final_top_k=4,
        lambda_param=0.7
    )
    
    return mmr_chunks 


