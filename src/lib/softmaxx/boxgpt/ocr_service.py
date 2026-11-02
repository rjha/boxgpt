import logging 
from pathlib import Path
from mistralai.client import Mistral



class MistralStub:
    def __init__(self, api_key: str):
        """Initializes the Mistral client with the provided API key."""
        self.client = Mistral(api_key=api_key)
        
    def process(self, source_path: Path, target_path: Path, include_blocks=True) -> None:
        logger = logging.getLogger("main." + __name__)

        if not source_path:
            raise ValueError("source path is required")

        if not target_path:
            raise ValueError("target path is required")
        
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
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url
            },
            include_blocks=include_blocks
        )

        logger.info("OCR response received...")
        ocr_response_dump = ocr_response.model_dump_json(indent=2)
        
        # Ensure output directory exists and save response
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(ocr_response_dump)
            
        logger.info(f"OCR response saved to {target_path}")


