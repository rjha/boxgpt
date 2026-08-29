from pathlib import Path
import json
from typing import List, Dict, Any



def process_ocr_dump(file_path: Path) -> List[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    
    pages = ocr_data.get("pages", [])
    structured_sections: List[Dict[str, Any]] = []
    
    current_section: Dict[str, Any] = {
        "section_title": "Preamble / Document Start",
        "page_start": 1,
        "page_end": 1,
        "content_blocks": []
    }

    # Define structural categories
    IGNORED_TYPES = {"image", "header", "footer"}
    CONTENT_TYPES = {"text", "list", "table", "code", "equation", "caption"}

    for page in pages:
        page_num = page.get("index", 0) + 1
        blocks = page.get("blocks", [])

        for block in blocks:
            b_type = block.get("type")
            content = (block.get("content") or block.get("text") or "").strip()

            if not content or b_type in IGNORED_TYPES:
                continue

            # Check if block triggers a section split
            is_title_block = b_type == "title"
            is_markdown_header = b_type == "text" and content.startswith("#")

            if is_title_block or is_markdown_header:
                # 1. Flush the active section if it contains collected blocks
                if current_section["content_blocks"]:
                    structured_sections.append({
                        "section_title": current_section["section_title"],
                        "page_range": [current_section["page_start"], current_section["page_end"]],
                        "content": "\n\n".join(current_section["content_blocks"]).strip()
                    })

                # 2. Open new section container
                current_section = {
                    "section_title": content.lstrip("#").strip(),
                    "page_start": page_num,
                    "page_end": page_num,
                    "content_blocks": []
                }

            # Accumulate content for text, list, table, etc.
            elif b_type in CONTENT_TYPES:
                current_section["content_blocks"].append(content)
                current_section["page_end"] = page_num

    # Flush final section
    if current_section["content_blocks"]:
        structured_sections.append({
            "section_title": current_section["section_title"],
            "page_range": [current_section["page_start"], current_section["page_end"]],
            "content": "\n\n".join(current_section["content_blocks"]).strip()
        })

    return structured_sections


def do_processing():
    ocr_file = Path("./doc01_ocr_response.json")
    content_file = Path("./doc01_ocr_content.json")
    if not ocr_file.exists() or not ocr_file.is_file():
        raise FileNotFoundError(f"Missing file: {ocr_file}")
    
    structured_sections = process_ocr_dump(ocr_file)
    json_string = json.dumps(structured_sections, indent=4, ensure_ascii=False)

    with content_file.open("w", encoding="utf-8") as f:
        f.write(json_string)
        print("OCR content saved")


if __name__ == "__main__":
    do_processing()
