import logging 
import re
import time 
from pathlib import Path
from mistralai.client import Mistral
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MistralClientConfig:
    network_delay: float = 1.0
    include_blocks: bool = True
    log_level: int = 1
    dial_mistral: bool = True


def _get_canonical_snake_name(file_stem: str) -> str:
    """
    Converts a file stem to strict snake_case:
    - Converts to lowercase
    - Replaces spaces and special characters with underscores
    - Collapses multiple underscores into a single one
    """
    # 1. Convert name to lowercase
    # 2. Replace NOT [letter, number, _] with underscore
    # 3. Strip leading/trailing underscores
    cleaned = file_stem.lower()
    snake = re.sub(r'[^a-z0-9_]+', '_', cleaned)
    return snake.strip('_')


class MistralClient:
    def __init__(self, api_key: str):
        """Initializes the Mistral client with the provided API key."""
        self.client = Mistral(api_key=api_key)
    
    def process(
        self, 
        source_path: Path, 
        target_path: Path, 
        config: Optional[MistralClientConfig] = None
    ) -> None:

        logger = logging.getLogger("main." + __name__)
        api_config = config or MistralClientConfig()  

        resolved_path = source_path.expanduser().resolve()
        logger.info(f"Uploading {resolved_path.name} to Mistral API...")
        
        # Upload the document to get a temp URL
        with open(resolved_path, "rb") as f:
            uploaded_file = self.client.files.upload(
                file={"file_name": resolved_path.name, "content": f},
                purpose="ocr"
            )

        logger.info(f"doc_path: {resolved_path}, uploaded_file_id: {uploaded_file.id}")
        
        # Obtain signed URL for OCR processing
        signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)   
        logger.info(f"doc_path: {resolved_path}, ocr_signed_url: {signed_url.url}")

        # Run Mistral OCR
        api_response_obj = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url
            },
            include_blocks=api_config.include_blocks
        )
        
        logger.info("OCR api response received...")
        ocr_content = api_response_obj.model_dump_json(indent=2)

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(ocr_content)
            
        logger.info(f"OCR api response saved to {target_path}")



    def process_batch(
        self,
        source_dir: Path, 
        target_dir: Path, 
        config: Optional[MistralClientConfig] = None
    ):

        logger = logging.getLogger("main." + __name__)
        # input check: source_dir and target_dir
        if not (source_dir.exists() and source_dir.is_dir()):
            raise ValueError("source directory not found")

        if not (target_dir.exists() and target_dir.is_dir()):
            raise ValueError("target directory not found")

        api_config = config or MistralClientConfig() 
        # gather all source files from source_dir
        pdf_files = sorted(list(source_dir.rglob("*.pdf")))
        
        for pdf_file in pdf_files:
            logger.info(f"source pdf_file is {pdf_file}")
            ocr_target_name = _get_canonical_snake_name(pdf_file.stem)
            ocr_target_path = target_dir / f"{ocr_target_name}.json"
            logger.info(f"OCR target path is {ocr_target_path}")

            # skip if already processed
            if ocr_target_path.exists():
                logger.info(f"skipping processed OCR file: {ocr_target_path.name}")
                continue
            
            # send to mistral API 
            if api_config.dial_mistral:
                self.process(pdf_file, ocr_target_path, api_config)
            
            logger.info(f"sleeping for {api_config.network_delay} seconds...\n")
            time.sleep(api_config.network_delay)