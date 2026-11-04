import signal 
import sys
import time
import logging 
import json
from pathlib import Path
from softmaxx.config import AppConfig
import pp.ab.smp as ab_smp


def signal_handler(sig, frame):
    print("\n[!] aborting on keyboard press...")
    sys.exit(130)


def process_batch(source_dir: Path, target_dir: Path):
    logger = logging.getLogger("main." + __name__)
    # input check: source_dir and target_dir
    if not (source_dir.exists() and source_dir.is_dir()):
        raise ValueError("source directory not found")

    if not (target_dir.exists() and target_dir.is_dir()):
        raise ValueError("target directory not found")

    
    # gather all source files from source_dir
    ocr_files = sorted(list(source_dir.rglob("*.json")))
    
    for ocr_file in ocr_files:
        print(f"source ocr_file is {ocr_file}")
        target_path = target_dir / f"{ocr_file.stem}.md"
        print(f"preprocessor output path is {target_path}")

        # skip if already processed
        if target_path.exists():
            logger.info(f"skipping pre-processed file: {target_path.name}")
            continue

        with open(ocr_file, "r", encoding="utf-8") as f:
            ocr_content = json.load(f)
        
        pp_content = ab_smp.process_doc(ocr_data=ocr_content)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(pp_content)
            
        logger.info(f"preprocessed file saved to {target_path}")
        time.sleep(1)


                
def main():
   
    print("run ocr/mai_batch.py script...")
    print(" ########### Press CTRL+C to abort ##########")
    time.sleep(1)

    AppConfig.load()
    log_config = AppConfig.get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    # @IMP: check SRC and TARGET_DIR
    ROOT_DIR = Path('/Users/rjha/code/github/boxgpt/scripts/temp').expanduser()
    SRC_DIR = ROOT_DIR / "ocr"
    TARGET_DIR = ROOT_DIR / "pp"
    process_batch(SRC_DIR, TARGET_DIR)



if __name__ == "__main__":
    # Register the SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)
    main()
