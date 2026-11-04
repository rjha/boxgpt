import json 
from pathlib import Path
import pp.ab.smp as ab_smp

# Add "/Users/rjha/code/github/boxgpt/scripts" to PYTHONPATH 
# Before running this script on the terminal. That is needed 
# to use prep.sb.smp import 

if __name__ == "__main__":
    # @IMP: check OCR_DIR and FILE
    SRC_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()
    TARGET_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()
    OCR_FILE = "mai_ocr.json"
    DOC_CATEGORY = "SOP"

    # No changes below this
    ocr_source_path = SRC_DIR / OCR_FILE
    print(ocr_source_path)
    with open(ocr_source_path, "r", encoding="utf-8") as f:
        ocr_content = json.load(f)

    result = ab_smp.process_doc(DOC_CATEGORY, ocr_content)
    # write preprocessed content 
    target_path = TARGET_DIR / "ab_ocr_pp.md"
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(result)

    

