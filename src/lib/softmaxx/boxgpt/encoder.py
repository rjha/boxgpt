
from typing import Protocol
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
from softmaxx.boxgpt.client.llm import DigitalOceanClient


@dataclass(frozen=True)
class EncoderConfig:
    batch_size: int = 64
    normalize_embeddings: bool = True
    show_progress_bar: bool = False


class TextEncoder(Protocol):
    @property
    def config(self)-> EncoderConfig:
            ...

    def encode(self, texts: list[str]) -> np.ndarray:
        ...
    

class DOBGEM3Encoder:
    EMBEDDING_DIM = 1024
    def __init__(self, 
        api_client:DigitalOceanClient, 
        config:EncoderConfig = EncoderConfig()
    ):
        self._api_client = api_client
        self._config = config

    @property
    def config(self) -> EncoderConfig:
        return self._config 
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        get embeddings using BAAI/bge-m3 model from DO Inference
        """
        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        if len(texts) > self._config.batch_size:
            raise ValueError(
                f"Input length ({len(texts)}) exceeds configured batch_size "
                f"({self._config.batch_size})."
            )
        
        api_response = self._api_client.bgem3_embeddings(texts)
        sorted_items = sorted(api_response.data, key=lambda item: item.index)
        emb_matrix = np.array(
            [item.embedding for item in sorted_items],
            dtype=np.float32,
        )

        # L2-normalize to match normalize_embeddings=True (Euclidean norm = 1.0)
        # Normalizing simplifies cosine distance calculations
        if self._config.normalize_embeddings:
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            emb_matrix = emb_matrix / np.maximum(norms, 1e-12)
        
        return emb_matrix
    

class OpenAIBGEM3Encoder:
    """
    An adapter that satisfies the TextEncoder interface contract,
    allowing box_rag to interact directly with the native OpenAI client.
    """
    EMBEDDING_DIM = 1024

    def __init__(self, api_client: OpenAI, config: EncoderConfig = EncoderConfig()):
        self._api_client = api_client
        self._config = config

    @property
    def config(self) -> EncoderConfig:
        return self._config 

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        if len(texts) > self._config.batch_size:
            raise ValueError(
                f"Input length ({len(texts)}) exceeds configured batch_size "
                f"({self._config.batch_size})."
            )
        
        # Interacting purely via the native standard call structure
        api_response = self._api_client.embeddings.create(
            input=texts,
            model="bge-m3",
            encoding_format="float"
        )
        
        # Maintain positioning index stability
        sorted_items = sorted(api_response.data, key=lambda item: item.index)
        emb_matrix = np.array(
            [item.embedding for item in sorted_items],
            dtype=np.float32,
        )

        # L2-normalization loop matching baseline expectations
        if self._config.normalize_embeddings:
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            emb_matrix = emb_matrix / np.maximum(norms, 1e-12)
        
        return emb_matrix

    