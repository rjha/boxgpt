import logging
import numpy as np 

# softmaxx libraries
from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.client.llm import DigitalOceanClient
from softmaxx.boxgpt.encoder import (
    DOBGEM3Encoder,
    LocalBGEM3Encoder,
    TextEncoder
)


def local_encoding(text: str) -> np.ndarray:
    text_encoder: TextEncoder = LocalBGEM3Encoder(model_path="/home/rjha/code/models/bge-m3")
    emb_matrix = text_encoder.encode([text])
    return emb_matrix

def do_api_encoding(text: str) -> np.ndarray:
    api_keys = AppConfig.get("api_keys")
    do_client = DigitalOceanClient(api_key=api_keys["DO_INFERENCE_KEY"])

    text_encoder: TextEncoder = DOBGEM3Encoder(api_client=do_client)
    emb_matrix = text_encoder.encode([text])
    return emb_matrix


def main():
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)
    logger = logging.getLogger("main." + __name__)

    USER_QUERY = "what are the PPE requirements for compressor maintenance?"
    local_emb_matrix = local_encoding(USER_QUERY)
    do_emb_matrix = do_api_encoding(USER_QUERY)
    
    # 3. Compare vectors (Cosine similarity of normalized vectors = dot product)
    cosine_sims = np.sum(local_emb_matrix * do_emb_matrix, axis=1)
    max_abs_diff = np.max(np.abs(local_emb_matrix - do_emb_matrix))

    print(f"Cosine similarities (1.0 = identical): {cosine_sims}")
    print(f"Max element-wise difference: {max_abs_diff:.6f}")

    assert local_emb_matrix.shape == do_emb_matrix.shape, (
        f"Shape mismatch: local {local_emb_matrix.shape} vs API {do_emb_matrix.shape}"
    )
    assert np.all(cosine_sims >= 0.995), (
        f"Embedding similarity below threshold (0.995): {cosine_sims}"
    )
    
if __name__ == "__main__":
    main()