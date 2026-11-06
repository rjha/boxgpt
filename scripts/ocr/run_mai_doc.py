from pathlib import Path
from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.client.ocr import MistralClient 


def main():
   
    print("run ocr/mai_test.py script...")
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    api_keys = AppConfig.get("api_keys")
    mistral_api_key = api_keys["MISTRAL"]
    ocr_client = MistralClient(api_key=mistral_api_key)
    
    # @IMP: change source directory and file name
    SRC_DIR = Path('~/backup/ab-docs/sop').expanduser()
    SRC_FILE = "sample01.pdf"

    # verify target folder 
    TARGET_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()

    # No changes below this
    source_path = SRC_DIR /  SRC_FILE
    target_path = TARGET_DIR / "mai_ocr.json"
    ocr_client.process(source_path=source_path, target_path=target_path)
    

if __name__ == "__main__":
    main()