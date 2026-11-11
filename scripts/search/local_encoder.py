from sentence_transformers import SentenceTransformer
from softmaxx.boxgpt.encoder import EncoderConfig

"""
The local BAAI/BGE-m3 encoder pulls in massive dependencies 
and can overwhelm the memory of small VM. If we start 4 fastapi 
workers with sentence transformers then it will easily use 
4 x 700 MB. Hence we should keep the local encoders and models 
separate from API and library paths.
"""
class LocalBGEM3Encoder:
    EMBEDDING_DIM = 1024
    def __init__(self, model_path:str, config:EncoderConfig = EncoderConfig()):
        self.model_instance = SentenceTransformer(model_path)
        self._config = config

    @property
    def config(self) -> EncoderConfig:
        return self._config 

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encodes chunk texts in batches using BAAI/bge-m3 and attaches 
        1024-dimensional float arrays to each chunk object.
        """
        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        # Normalizing simplifies cosine distance calculations
        # local models response do not need sorting as they return 
        # the embeddings using the exact input sequence
        embeddings = self.model_instance.encode(
            texts, 
            batch_size=self._config.batch_size, 
            show_progress_bar=self._config.show_progress_bar,
            normalize_embeddings=self._config.normalize_embeddings  
        )
        
        return embeddings

