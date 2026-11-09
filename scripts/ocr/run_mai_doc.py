from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.client.ocr import MistralClient 
from scripts.constants import XOUT_DIR, DOC_DIR

def main():
   
    print("run ocr/mai_test.py script...")
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    api_keys = AppConfig.get("api_keys")
    mistral_api_key = api_keys["MISTRAL"]
    ocr_client = MistralClient(api_key=mistral_api_key)
    
    # No changes below this 
    source_path = DOC_DIR / "sample"/ "sample.pdf"
    target_path = XOUT_DIR / "ocr" / "sample_ocr.json"
    ocr_client.process(source_path=source_path, target_path=target_path)
    

if __name__ == "__main__":
    main()