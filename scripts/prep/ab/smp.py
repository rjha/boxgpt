import json
import re
from pathlib import Path
from typing import Dict, Any, List


def parse_markdown_table_to_ppe(table_content: str) -> List[Dict[str, Any]]:
    """
    Parses a markdown table block, identifies if it's a PPE matrix, 
    and maps each activity to its checked/required PPE items.
    """
    lines = [line.strip() for line in table_content.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        return []

    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    if not headers:
        return []

    ppe_mappings = []
    ppe_headers = headers[1:]  # Skipping the activity column header

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
    """
    Pre-processes raw OCR JSON output using blocks, converts PPE tables 
    into clean structured text for the content stream, and builds a unique PPE list.
    """
    pages = ocr_data.get("pages", [])
    if not pages:
        return {"metadata_hints": {}, "ppe_matrices": [], "unique_ppe_list": [], "clean_text": ""}

    extracted_metadata = {}
    extracted_ppe_matrices = []
    unique_ppe_set = set()
    cleaned_page_texts: List[str] = []

    for page in pages:
        page_index = page.get("index", 0)
        blocks = page.get("blocks", [])
        page_content_lines = []
        # Turn off at page level 
        ppe_flag = False 
        
        for block in blocks:
           
            b_type = block.get("type", "")
            content = block.get("content", "").strip()

            # exclude certain type of content 
            if not content or b_type == "image":
                continue 
            # drop header & footer 
            if b_type in ["header", "footer"] and (page_index > 0):
                continue

            # Drop confidential notices
            if "Company Confidential Document" in content or re.search(r"Page \d+ of \d+", content):
                continue

            # process PPE Matrices
            if not ppe_flag and b_type == "title" and "PPE" in content:
                ppe_flag = True 
                page_content_lines.append(content)
                continue 
                
            if page_index == 0:
                if b_type in ["title", "header", "text", "table"] and content:
                    if "Document" in content or "Title" in content or "SMP" in content:
                        if "header_summary" not in extracted_metadata:
                            extracted_metadata["header_summary"] = content
                        else:
                            extracted_metadata["header_summary"] += f" | {content}"

            
            if b_type == "table" and "PPE" in content:
                ppe_flag = True
            
            if ppe_flag:
                # print(f"inside PPE parsing, content: {content}")
                ppe_rows = parse_markdown_table_to_ppe(content)
                if ppe_rows:
                    extracted_ppe_matrices.extend(ppe_rows)
                    
                    # Aggregate unique PPE items for the separate list
                    for row in ppe_rows:
                        for ppe in row.get("required_ppe", []):
                            unique_ppe_set.add(ppe)
                    
                    # Convert table into a clean, structured text format and append it
                    converted_lines = ["### PPE Requirements by Activity"]
                    for row in ppe_rows:
                        ppe_str = ", ".join(row["required_ppe"])
                        converted_lines.append(f"- **{row['activity']}**: {ppe_str}")

                    page_content_lines.append("\n".join(converted_lines))
                    ppe_flag = False 
                    # continue
              
            # special cases done. append the content
            page_content_lines.append(content)

        if page_content_lines:
            page_text = "\n".join(page_content_lines)
            cleaned_page_texts.append(f"--- Page {page_index + 1} ---\n{page_text}")

    full_clean_document = "\n\n".join(cleaned_page_texts)

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
        
    