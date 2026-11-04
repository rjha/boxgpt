from pathlib import Path
from softmaxx.config import AppConfig
from softmaxx.boxgpt.ocr_service import MistralStub 


def main():
   
    print("run ocr/mai_test.py script...")
    AppConfig.load()
    log_config = AppConfig.get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    api_keys = AppConfig.get("api_keys")
    mistral_api_key = api_keys["MISTRAL"]
    ocr_stub = MistralStub(api_key=mistral_api_key)
    
    # @IMP: change source directory and file name
    SRC_DIR = Path('~/Desktop/x-projects/AB-docs/SOP').expanduser()
    SRC_FILE = "SOP for AFR Liquid handling, unloading, storage and co-processing.pdf"

    # verify target folder 
    TARGET_DIR = Path('~/code/github/boxgpt/scripts/temp').expanduser()

    # No changes below this
    source_path = SRC_DIR /  SRC_FILE
    target_path = TARGET_DIR / "mai_ocr_response.json"
    ocr_stub.process(source_path=source_path, target_path=target_path)
    

if __name__ == "__main__":
    main()