import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set


def _extract_page_zero_metadata(b_type: str, content: str) -> str:
    """Helper function that returns metadata content if criteria are met, else an empty string."""
    if b_type in ["title", "header", "text", "table"] and content:
        if "Document" in content or "Title" in content or "SMP" in content:
            return content
    return ""

def _is_ppe_table(table_content: str) -> bool:
    """Checks if a table is a PPE matrix based on headers."""
    lines = [line.strip() for line in table_content.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        return False

    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    if not headers:
        return False

    header_str = " ".join(headers).lower()
    ppe_keywords = ["helmet", "goggles", "shoe", "gloves", "mask", "ppe", "jacket", "shield"]
    return any(kw in header_str for kw in ppe_keywords)


def _parse_ppe_table(table_content: str) -> List[Dict[str, Any]]:
    """Parses a verified PPE table into structured rows."""
    lines = [line.strip() for line in table_content.strip().split("\n") if line.strip()]
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    
    ppe_mappings = []
    ppe_headers = headers[1:]

    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < len(headers):
            continue
        
        activity_name = cells[0]
        required_ppes = []
        
        for header_name, cell_value in zip(ppe_headers, cells[1:]):
            if cell_value in ["√", "✓", "x", "X"] or cell_value.lower() in ["yes", "true"]:
                required_ppes.append(header_name)
                
        if activity_name:
            ppe_mappings.append({
                "activity": activity_name,
                "required_ppe": required_ppes
            })

    return ppe_mappings


def preprocess_ocr_json(ocr_data: Dict[Any, Any]) -> Dict[str, Any]:
    pages = ocr_data.get("pages", [])
    if not pages:
        return {
            "metadata_hints": {}, 
            "ppe_matrices": [], 
            "unique_ppe_list": [], 
            "clean_text": ""
        }

    header_items = []
    extracted_ppe_matrices = []
    unique_ppe_set = set()
    cleaned_page_texts: List[str] = []

    for page in pages:
        page_index = page.get("index", 0)
        blocks = page.get("blocks", [])
        page_content_lines = []
        

        for block in blocks:
            b_type = block.get("type", "")
            content = block.get("content", "").strip()

            # specials cases skip processing 
            if not content or b_type == "image":
                continue 
            
            if b_type in ["header", "footer"] and (page_index > 0):
                continue

            if "Company Confidential Document" in content or re.search(r"Page \d+ of \d+", content):
                continue

            # Capture metadata on Page 0 cleanly into a list
            if page_index == 0:
                meta_piece = _extract_page_zero_metadata(b_type, content)
                if meta_piece:
                    header_items.append(meta_piece)

            if b_type == "table" and _is_ppe_table(content):
                ppe_rows = _parse_ppe_table(content)
                if ppe_rows:
                    extracted_ppe_matrices.extend(ppe_rows)
                    for row in ppe_rows:
                        for ppe in row.get("required_ppe", []):
                            unique_ppe_set.add(ppe)

            # default content
            page_content_lines.append(content)

        if page_content_lines:
            page_text = "\n".join(page_content_lines)
            cleaned_page_texts.append(f"--- Page {page_index + 1} ---\n{page_text}")

    full_clean_document = "\n\n".join(cleaned_page_texts)

    # Build final metadata dict with header_summary guaranteed to be present
    extracted_metadata = {
        "header_summary": " | ".join(header_items) if header_items else ""
    }

    return {
        "metadata_hints": extracted_metadata,
        "ppe_matrices": extracted_ppe_matrices,
        "unique_ppe_list": sorted(list(unique_ppe_set)),
        "clean_text": full_clean_document
    }



if __name__ == "__main__":
    ocr_output_path = Path('~/code/github/boxgpt/docs/ocr/ab_smp_tpp_om_014.json').expanduser()
    with open(ocr_output_path, "r", encoding="utf-8") as f:
        ocr_json = json.load(f)

    result = preprocess_ocr_json(ocr_json)
    print(result["clean_text"])
    
    # print(result["metadata_hints"])
    # print(json.dumps(result["ppe_matrices"], indent=2))
    # print(json.dumps(result["unique_ppe_list"], indent=2))
        
    