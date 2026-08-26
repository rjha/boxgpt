
import os
from pathlib import Path
# from mistralai import Mistral
from mistralai.client import Mistral

# 1. Initialize API Client
# Set MISTRAL_API_KEY in your environment or pass it directly
client = Mistral(api_key="xxxxxxx")

def ingest_pdf_with_mistral_ocr(pdf_path: str) -> str:
    print(f"Uploading {pdf_path} to Mistral API...")
    
    # Upload the PDF file to get a temporary processing URL
    with open(pdf_path, "rb") as f:
        uploaded_file = client.files.upload(
            file={"file_name": os.path.basename(pdf_path), "content": f},
            purpose="ocr"
        )
    
    # Obtain signed URL for OCR processing
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
    
    print(f"file uploaded, mistral signed URL is {signed_url}")
    # 2. Run Mistral OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url
        }
    )

    print("OCR response received...")
    # 3. Aggregate all page Markdown outputs into a single document string
    full_markdown = []
    for page in ocr_response.pages:
        full_markdown.append(f"<!-- Page {page.index + 1} -->\n{page.markdown}")
        
    return "\n\n".join(full_markdown)

# Example Execution
pdf_path = Path('~/Downloads/ICAR_MAG/Jan-Feb_2026.pdf').expanduser()
markdown_doc = ingest_pdf_with_mistral_ocr(pdf_path)

# Save extracted text locally for the chunking step
with open("icar_mag.md", "w", encoding="utf-8") as out:
    out.write(markdown_doc)

print("Ingestion Complete! Output saved to icar_mag.md")