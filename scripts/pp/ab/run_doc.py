import json 
import scripts.pp.ab.smp as ab_smp 
from scripts.constants import XOUT_DIR, DOC_DIR


if __name__ == "__main__":
   
    DOC_CATEGORY = "SWP"
    ocr_source_path = DOC_DIR / "sample" / "ocr" / "sample_ocr.json"
    print(ocr_source_path)
    with open(ocr_source_path, "r", encoding="utf-8") as f:
        ocr_content = json.load(f)

    result = ab_smp.process_doc(DOC_CATEGORY, ocr_content)
    # write preprocessed content 
    target_path = XOUT_DIR / "pp" / "sample_ocr_pp.md"
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(result)

    

