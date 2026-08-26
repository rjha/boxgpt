from pathlib import Path
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# 1. Set EasyOCR with Devanagari ('hi') and English ('en')
ocr_options = EasyOcrOptions(
    lang=["hi", "en"],
    force_full_page_ocr=True  # Renders PDF pages as images to bypass bad fonts
)

# 2. Configure Docling Pipeline
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = ocr_options
pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# 3. Instantiate Document Converter
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# 4. Convert Document
pdf_path = "/home/rjha/Downloads/Jan-Feb_2026_10.pdf"
result = converter.convert(pdf_path)

# 5. Save the output to disk
output_file = Path("easy_ocr_out.md")
output_file.write_text(result.document.export_to_markdown(), encoding="utf-8")

print(f"Success! Visual OCR completed and saved to: {output_file.resolve()}")