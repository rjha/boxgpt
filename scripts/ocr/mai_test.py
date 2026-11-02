from pathlib import Path
from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.ocr_service import MistralStub 


def main():
   
    print("run ocr/mai_test.py script...")
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    api_keys = AppConfig.get("api_keys")
    mistral_api_key = api_keys["MISTRAL"]

    # Instantiate the class with the API key
    ocr_stub = MistralStub(api_key=mistral_api_key)
    
    # Example Execution
    SRC_DIR = Path('/Users/rjha/Desktop/x-projects/AB-docs/SMP/TPP').expanduser()
    SCRIPT_TEMP_DIR = Path('/Users/rjha/code/github/boxgpt/scripts/temp').expanduser()
    source_path = SRC_DIR /  "SMP TPP O&M 016 Maintenance of SWAS Panel.pdf"
    target_path = SCRIPT_TEMP_DIR / "ocr_response.json"
    ocr_stub.process(source_path=source_path, target_path=target_path)
    

if __name__ == "__main__":
    main()