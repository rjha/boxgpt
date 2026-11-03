import signal 
import sys
import time
from pathlib import Path
from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.ocr_service import MistralStub, MistralStubConfig 



def signal_handler(sig, frame):
    print("\n[!] aborting on keyboard press...")
    sys.exit(130)


def main():
   
    print("run ocr/mai_batch.py script...")
    print(" ########### Press CTRL+C to abort ##########")
    time.sleep(4)

    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    api_keys = AppConfig.get("api_keys")
    mistral_api_key = api_keys["MISTRAL"]

    # Instantiate the class with the API key
    ocr_stub = MistralStub(api_key=mistral_api_key)
    
    # get all SMP documents and write to script/temp/ocr folder
    SRC_DIR = Path('/Users/rjha/Desktop/x-projects/AB-docs/SMP').expanduser()
    ROOT_DIR = Path('/Users/rjha/code/github/boxgpt/scripts/temp').expanduser()
    OCR_DIR = ROOT_DIR / "ocr"
    mistral_config: MistralStubConfig = MistralStubConfig(
        dial_mistral=False,
        network_delay=5.0)
    
    ocr_stub.process_batch(SRC_DIR, OCR_DIR, mistral_config)
    

if __name__ == "__main__":
    # Register the SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)
    main()
