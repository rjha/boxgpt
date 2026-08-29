
import os
import logging 
from pathlib import Path
from mistralai.client import Mistral
from softmaxx.config import AppConfig, get_logger_config


client = None
logger = logging.getLogger("main." + __name__)

def ingest_pdf_with_mistral_ocr(pdf_path: str) -> str:
    global client 
    print(f"Uploading {pdf_path} to Mistral API...")
    
    # Upload the PDF file to get a temporary processing URL
    with open(pdf_path, "rb") as f:
        uploaded_file = client.files.upload(
            file={"file_name": os.path.basename(pdf_path), "content": f},
            purpose="ocr"
        )

    doc_path = pdf_path.resolve()
    print(f"doc_path: {doc_path}, uploaded_file_id: {uploaded_file.id}")
    
    # Obtain signed URL for OCR processing
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id)   
    print(f"doc_path: {doc_path}, ocr_signed_url: {signed_url}")

    # 2. Run Mistral OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url
        },
        include_blocks=True
    )

    print("OCR response received...")
    ocr_response_dump = ocr_response.model_dump_json(indent=2)
    with open("ocr_response.json", "w", encoding="utf-8") as f:
        f.write(ocr_response_dump)
    print("OCR response saved")

def process_doc():
    global client 
    print("run scan OCR4 script...")

    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    api_keys = AppConfig.get("api_keys")
    mistral_api_key = api_keys["mistral"]
    client = Mistral(api_key=mistral_api_key)

    # Example Execution
    pdf_path = Path('~/Downloads/Jan-Feb_2026.pdf').expanduser()
    ingest_pdf_with_mistral_ocr(pdf_path)


if __name__ == "__main__":
    process_doc()