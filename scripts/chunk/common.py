import json

def transform_slug(doc_slug: str) -> str:
    readable_text = doc_slug.replace("_", " ").replace("-", " ")
    title_str = readable_text.title()
    
    # Optional: Normalize common industrial acronyms
    acronyms = {"Lt": "LT", "Smp": "SMP", "Sop": "SOP", "Hvac": "HVAC"}
    for old, new in acronyms.items():
        title_str = title_str.replace(old, new)
        
    return title_str


def save_json_to_file(content, target_path):
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def read_json_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"unable to locate file: {file_path}")

    # Load JSON content from file
    with file_path.open("r", encoding="utf-8") as f:
        content = json.load(f)

    return content 
    