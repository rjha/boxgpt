import signal 
import sys
import time
from pathlib import Path
from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.ocr_service import MistralStub, MistralStubConfig 



def signal_handler(sig, frame):
    print("\n[!] aborting on keyboard press...")
    sys.exit(130)

def is_dir_empty(path):
    dir_path = Path(path)
    # next() returns None if iterdir() is empty
    return next(dir_path.iterdir(), None) is None


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
    ocr_stub = MistralStub(api_key=mistral_api_key)
    
    # @IMP: check source folder and 
    SRC_DIR = Path('/Users/rjha/Desktop/x-projects/AB-docs/SOP').expanduser()
    ROOT_DIR = Path('/Users/rjha/code/github/boxgpt/scripts/temp').expanduser()
    TARGET_DIR = ROOT_DIR / "ocr"

    # Target DIR should be empty for batch Jobs 
    # if not is_dir_empty(TARGET_DIR):
    #    raise ValueError(f"Abort: Target directory {TARGET_DIR} is not empty!!")

    # @IMP: when dial_mistral=False, the batch processing will not 
    # call mistral API at all. dial_mistral can be used for testing 
    # the directory walkthrough without calling API.
    mistral_config: MistralStubConfig = MistralStubConfig(
        dial_mistral=True,
        network_delay=5.0)
    
    ocr_stub.process_batch(SRC_DIR, TARGET_DIR, mistral_config)
    

if __name__ == "__main__":
    # Register the SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)
    main()
