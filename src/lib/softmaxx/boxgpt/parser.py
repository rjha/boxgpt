import re
from typing import Any, Literal



def _build_chunks(
    title: str, 
    content: str, 
    page_range: list[int], 
    max_chars: int, 
    overlap: int
) -> list[dict[str, Any]]:
    """Enforces max character caps and creates sliding window sub-chunks with overlap."""
    if len(content) <= max_chars:
        return [{
            "section_title": title,
            "page_range": page_range,
            "content": f"### Location: {title}\n\n{content}"
        }]

    sub_chunks = []
    start = 0
    chunk_idx = 1
    
    while start < len(content):
        end = start + max_chars
        sub_text = content[start:end]
        
        # Clean paragraph boundary snap if near the end of the slice
        if end < len(content):
            last_newline = sub_text.rfind("\n\n")
            if last_newline > max_chars // 2:
                end = start + last_newline
                sub_text = content[start:end]

        sub_chunks.append({
            "section_title": f"{title} (Part {chunk_idx})",
            "page_range": page_range,
            "content": f"### Location: {title}\n\n{sub_text.strip()}"
        })
        
        start = end - overlap if end < len(content) else len(content)
        chunk_idx += 1

    return sub_chunks



def _infer_sections_fsm(
    markdown_text: str, 
    max_chars: int,
    overlap: int
) -> list[dict[str, Any]]:
    """
    State machine parser for clean Markdown documents.
    States: 
      - STATE_DEFAULT: Accumulating text, watching for headers, page markers, or code fences.
      - STATE_CODE: Inside a code block, ignoring all header and page syntax.
    """
    # Define States
    STATE_DEFAULT = "DEFAULT"
    STATE_CODE = "CODE"
    
    current_state = STATE_DEFAULT
    
    sections = []
    header_stack = {} # Tracks breadcrumbs: {1: "H1", 2: "H2", 3: "H3"}
    current_page_start = 1
    current_page_end = 1
    current_lines = []

    def get_header_path() -> str:
        if not header_stack:
            return "Introduction"
        return " > ".join(header_stack[lvl] for lvl in sorted(header_stack.keys()))

    def flush_section():
        nonlocal current_lines
        content_text = "\n".join(current_lines).strip()
        if content_text:
            sections.extend(_build_chunks(
                title=get_header_path(),
                content=content_text,
                page_range=[current_page_start, current_page_end],
                max_chars=max_chars,
                overlap=overlap
            ))
            current_lines = []

    lines = markdown_text.splitlines()

    for line in lines:
        stripped = line.strip()

        # --- STATE TRANSITION: Code Fence Toggle ---
        if stripped.startswith("```"):
            if current_state == STATE_DEFAULT:
                current_state = STATE_CODE
            else:
                current_state = STATE_DEFAULT
            
            current_lines.append(line)
            continue

        # --- STATE-BASED PROCESSING ---
        if current_state == STATE_CODE:
            # Inside code block: treat everything as raw text, ignore headers/pages
            current_lines.append(line)
            continue

        # --- STATE_DEFAULT LOGIC ---
        
        # 1. Check for Page Marker Transition
        page_match = re.match(r"^---\s+Page\s+(\d+)\s+---", stripped, re.IGNORECASE)
        if page_match:
            current_page_end = int(page_match.group(1))
            continue

        # 2. Check for Header Transition
        header_match = re.match(r"^(#{1,4})\s+(.+)", stripped)
        if header_match:
            # Commit whatever we gathered before hitting this header
            flush_section()
            
            # Update header stack hierarchy
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            
            # Pop deeper levels when moving to a higher or equal header level
            header_stack = {lvl: t for lvl, t in header_stack.items() if lvl < level}
            header_stack[level] = title
            
            # Advance start page to current page position
            current_page_start = current_page_end
            continue

        # 3. Default text accumulation
        current_lines.append(line)

    # Flush final remaining content
    flush_section()
    return sections



def _infer_sections_regex(markdown_text: str) -> list[dict[str, Any]]:
    """
    Parses a cleaned markdown document into sections based on headers (#, ##, ###)
    and tracks approximate page numbers from '--- Page X ---' markers.
    """
    sections = []
    current_title = "Introduction"
    current_page = 1
    current_content_lines = []

    lines = markdown_text.splitlines()
    for line in lines:
        # Track page numbers if present in markdown
        page_match = re.match(r"^---\s+Page\s+(\d+)\s+---", line.strip(), re.IGNORECASE)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Detect markdown headers
        header_match = re.match(r"^(#{1,4})\s+(.+)", line.strip())
        if header_match:
            # If we already accumulated content for the previous section, save it
            if current_content_lines:
                sections.append({
                    "section_title": current_title,
                    "page_range": [current_page, current_page],
                    "content": "\n".join(current_content_lines).strip()
                })
                current_content_lines = []
            
            current_title = header_match.group(2).strip()
            continue

        current_content_lines.append(line)

    # Append the final section
    if current_content_lines:
        sections.append({
            "section_title": current_title,
            "page_range": [current_page, current_page],
            "content": "\n".join(current_content_lines).strip()
        })

    return sections


# #####################################
# PUBLIC METHODS 
#
#######################################


def infer_sections(
        markdown_text: str, 
        max_chars: int,
        overlap: int,
        strategy: Literal["regex", "fsm"]
    ) -> list[dict[str, Any]]:
    """Infers sections from markdown using requested strategy."""
    match strategy:
        case "regex":
            return _infer_sections_regex(markdown_text=markdown_text)
        case "fsm":
            return _infer_sections_fsm(
                markdown_text=markdown_text, 
                max_chars=max_chars, 
                overlap=overlap)
        case _:
            raise ValueError(f"unknown inference strategy {strategy}")        
