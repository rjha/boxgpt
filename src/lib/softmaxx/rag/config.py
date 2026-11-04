import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

# Step 2: Use existing AppConfig to get the path
from softmaxx.config import AppConfig


# Step 3: Domain-specific dataclasses encapsulate their own schema parsing
@dataclass(frozen=True)
class RetrievalConfig:
    doc_id: str
    candidate_limit: int
    rrf_k: int
    vector_weight: float
    lexical_weight: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalConfig":
        retrieval = data.get("retrieval", {})
        return cls(
            doc_id=str(retrieval.get("doc_id", "mai_out")),
            candidate_limit=int(retrieval.get("candidate_limit", 20)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            vector_weight=float(retrieval.get("vector_weight", 1.0)),
            lexical_weight=float(retrieval.get("lexical_weight", 1.0)),
        )


@dataclass(frozen=True)
class MMRConfig:
    final_top_k: int
    lambda_param: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MMRConfig":
        mmr = data.get("mmr", {})
        return cls(
            final_top_k=int(mmr.get("final_top_k", 4)),
            lambda_param=float(mmr.get("lambda_param", 0.7)),
        )


class RagConfig:
    """Static domain loader for RAG settings."""

    _raw_data: Optional[Dict[str, Any]] = None
    _is_loaded: bool = False

    @staticmethod
    def load(file_path: Optional[str | Path] = None) -> None:
        """
        Loads raw JSON once. 
        Resolves file path via AppConfig if not provided.
        """
        if RagConfig._is_loaded:
            return

        # Fetch file path from AppConfig mechanism if omitted
        if file_path:
            target_path = Path(file_path) 
        else:
            target_path = Path(AppConfig.get("rag_config_file"))

        with open(target_path, "r", encoding="utf-8") as f:
            RagConfig._raw_data = json.load(f)

        RagConfig._is_loaded = True

    @staticmethod
    def _get_raw() -> Dict[str, Any]:
        """Ensures load() is called before returning raw data."""
        RagConfig.load()
        return RagConfig._raw_data

    @staticmethod
    @lru_cache(maxsize=1)
    def get_retrieval_config() -> RetrievalConfig:
        """Isolated accessor for retrieval parameters."""
        return RetrievalConfig.from_dict(RagConfig._get_raw())

    @staticmethod
    @lru_cache(maxsize=1)
    def get_mmr_config() -> MMRConfig:
        """Isolated accessor for MMR parameters."""
        return MMRConfig.from_dict(RagConfig._get_raw())

    @staticmethod
    @lru_cache(maxsize=1)
    def get_system_prompt() -> str:
        """Isolated accessor for system prompt text."""
        raw = RagConfig._get_raw()
        prompt_path = Path(raw["system_prompt_path"])
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def clear_cache() -> None:
        """Invalidates raw data and clear all memoized accessors."""
        RagConfig._raw_data = None
        RagConfig._is_loaded = False
        RagConfig.get_retrieval_config.cache_clear()
        RagConfig.get_mmr_config.cache_clear()
        RagConfig.get_system_prompt.cache_clear()