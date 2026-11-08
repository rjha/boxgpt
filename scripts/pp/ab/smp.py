import re
from typing import Any


def _extract_page_zero_metadata(doc_category:str, b_type: str, content: str) -> str:
    """Helper function that returns metadata content if criteria are met, else an empty string."""
    if b_type in ["title", "header", "text", "table"] and content:
        if "Document" in content or "Title" in content or doc_category in content:
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


def _parse_ppe_table(table_content: str) -> list[dict[str, Any]]:
    """Parses a verified PPE table into structured rows."""
    lines = [line.strip() for line in table_content.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        return []
    
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    if not headers:
        return []
    
    ppe_mappings = []
    ppe_headers = headers[1:]

    # Skipping index 0 (headers) and index 1 (markdown separator like |---|---|)
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < len(headers):
            continue
        
        activity_name = cells[0]
        required_ppes = []
        
        for header_name, cell_value in zip(ppe_headers, cells[1:]):
            val_str = str(cell_value).lower()
            if cell_value in ["√", "✓", "x", "X"] or val_str in ["yes", "true"]:
                required_ppes.append(header_name)
                
        if activity_name:
            ppe_mappings.append({
                "activity": activity_name,
                "required_ppe": required_ppes
            })

    return ppe_mappings


def process_doc(doc_category:str, ocr_data: dict[Any, Any]) -> str:
    pages = ocr_data.get("pages", [])
    if not pages:
        return (
            "# DOCUMENT METADATA\n- **Header Summary:** N/A\n\n"
            "# DOCUMENT CONTENT\n"
            "# PPE MATRICES\nNone detected\n\n"
            "# UNIQUE PPE LIST\nNone\n\n"
        )

    header_items = []
    extracted_ppe_matrices = []
    unique_ppe_set = set()
    cleaned_page_texts: list[str] = []

    for page in pages:
        page_index = page.get("index", 0)
        blocks = page.get("blocks", [])
        page_content_lines = []
        

        for block in blocks:
            b_type = block.get("type", "")
            content = block.get("content", "").strip()

            # skip empty content or images 
            if not content or b_type == "image":
                continue 
            
            if b_type in ["header", "footer"] and (page_index > 0):
                continue

            if "Company Confidential Document" in content or re.search(r"Page \d+ of \d+", content):
                continue

            # Capture metadata on Page 0 cleanly into a list
            if page_index == 0:
                meta_piece = _extract_page_zero_metadata(doc_category, b_type, content)
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

    # Format Metadata Section
    header_summary = " | ".join(header_items) if header_items else "N/A"
    metadata_markdown = f"# DOCUMENT METADATA\n- **Header Summary:** {header_summary}"

    # Format PPE Matrices Section
    if extracted_ppe_matrices:
        ppe_lines = []
        for row in extracted_ppe_matrices:
            row_str = ", ".join(f"**{k}**: {v}" for k, v in row.items())
            ppe_lines.append(f"- {row_str}")
        ppe_markdown = "# PPE MATRICES\n" + "\n".join(ppe_lines)
    else:
        ppe_markdown = "# PPE MATRICES\n- None detected"

    # Format Unique PPE List Section
    unique_ppe_list = sorted(list(unique_ppe_set))
    unique_ppe_str = ", ".join(unique_ppe_list) if unique_ppe_list else "None"
    unique_ppe_markdown = f"# UNIQUE PPE LIST\n{unique_ppe_str}"

    # Format Clean Document Content Section
    full_clean_document = "\n\n".join(cleaned_page_texts)
    content_markdown = f"# DOCUMENT CONTENT\n\n{full_clean_document}"

    # Combine all sections into a single cohesive markdown document string
    return "\n\n---\n\n".join([
        metadata_markdown,
        content_markdown,
        ppe_markdown,
        unique_ppe_markdown
    ])