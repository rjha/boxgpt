
import os
from pathlib import Path
from mistralai.client import Mistral

# 1. Initialize API Client
# Set MISTRAL_API_KEY in your environment or pass it directly
# @todo read from config file 
client = Mistral(api_key="xxxxxx")

def ingest_pdf_with_mistral_ocr(pdf_path: str) -> str:
    print(f"Uploading {pdf_path} to Mistral API...")
    
    # Upload the PDF file to get a temporary processing URL
    with open(pdf_path, "rb") as f:
        uploaded_file = client.files.upload(
            file={"file_name": os.path.basename(pdf_path), "content": f},
            purpose="ocr"
        )

    print(f"file id is {uploaded_file.id}")
    
    # Obtain signed URL for OCR processing
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
    
    print(f"file uploaded, mistral signed URL is {signed_url}")
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

# Example Execution
pdf_path = Path('~/Downloads/Jan-Feb_2026_10.pdf').expanduser()
ingest_pdf_with_mistral_ocr(pdf_path)
