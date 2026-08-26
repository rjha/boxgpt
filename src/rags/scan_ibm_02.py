import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
    TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# 1. Configure Tesseract OCR options
ocr_options = TesseractOcrOptions(
    lang=["hin", "eng"],
    force_full_page_ocr=True  # Renders PDF pages as images to run visual OCR
)

# 2. Configure Pipeline Options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = ocr_options
pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# 3. Instantiate Converter
converter = DocumentConverter(
    format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
)

# 4. Process Document and Save Output
pdf_path = "/home/rjha/Downloads/Jan-Feb_2026.pdf"
result = converter.convert(pdf_path)

output_file = Path("tesseract_out.md")
output_file.write_text(result.document.export_to_markdown(), encoding="utf-8")

print(f"Success! Full-page Tesseract conversion saved to: {output_file.resolve()}")