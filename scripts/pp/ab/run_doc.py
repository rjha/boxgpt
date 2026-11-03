import json 
from pathlib import Path
from pp.ab.smp import process_doc

# Add "/Users/rjha/code/github/boxgpt/scripts" to PYTHONPATH 
# Before running this script on the terminal. That is needed 
# to use prep.sb.smp import 

if __name__ == "__main__":
    ROOT_DIR = Path('~/code/github/boxgpt/scripts/temp').expanduser()
    ocr_source_path = ROOT_DIR / "ocr" / "smpe_i006_oltc_maintenance_of_power_transformer.json"

    ocr_source_path = Path('~/code/github/boxgpt/docs/ocr/ab_smp_ei_005.json').expanduser()
    print(ocr_source_path)
    with open(ocr_source_path, "r", encoding="utf-8") as f:
        ocr_content = json.load(f)

    result = process_doc(ocr_content)
    print(result)
    

