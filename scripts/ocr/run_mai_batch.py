import signal 
import sys
import time
from pathlib import Path
from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.client.ocr import MistralClient, MistralClientConfig 


def signal_handler(sig, frame):
    print("\n[!] aborting on keyboard press...")
    sys.exit(130)


def main():
   

    print(" ##################################### ")
    print(" @@@  OCR BATCH SCRIPT             @@@ ")
    print(" @@@  Press CTRL+C to abort        @@@ ")
    print(" ##################################### ")

    time.sleep(2)
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file)

    api_keys = AppConfig.get("api_keys")
    mistral_api_key = api_keys["MISTRAL"]

    # Instantiate the class with the API key
    ocr_client = MistralClient(api_key=mistral_api_key)
    
    # @IMP: check source folder and 
    SRC_DIR = Path('~/backup/ab-docs/sop').expanduser()
    ROOT_DIR = Path('~/code/github/boxgpt/scripts/x.out').expanduser()
    TARGET_DIR = ROOT_DIR / "ocr"

    # @IMP: when dial_mistral=False, the batch processing will not 
    # call mistral API at all. dial_mistral can be used for testing 
    # the directory walkthrough without calling API.
    mistral_config: MistralClientConfig = MistralClientConfig(
        dial_mistral=True,
        network_delay=5.0)
    
    ocr_client.process_batch(SRC_DIR, TARGET_DIR, mistral_config)
    

if __name__ == "__main__":
    # Register the SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)
    main()
