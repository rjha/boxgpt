import json
import logging
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from softmaxx.config import AppConfig, get_logger_config


logger = logging.getLogger("main." + __name__)



@dataclass
class QueryPayload:
    smp_text_path: Path
    smp_schema_path: Path
    user_content_path: Path
    system_prompt_path: Path

    def _read_text_file(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path.read_text(encoding="utf-8")

    def get_system_prompt(self) -> str:
        return self._read_text_file(self.system_prompt_path)

    def get_user_content(self) -> str:
        document_text = self._read_text_file(self.smp_text_path)
        smp_schema = self._read_text_file(self.smp_schema_path)
        user_content_template = self._read_text_file(self.user_content_path)

        # This will be part of the smp_text 
        sample_metadata = {"header_summary": "Title: Battery bank | Doc: SMP/E&I/005"}
        sample_ppe = [{"activity": "Battery terminal cleaning", "required_ppe": ["Helmet", "Goggles"]}]
        
        llm_payload = {
            "metadata_hints": sample_metadata,
            "ppe_matrices": sample_ppe,
            "document_text": document_text
        }

        final_payload = json.dumps(llm_payload, indent=2)
        return user_content_template.replace("{{SCHEMA}}", smp_schema).replace("{{PAYLOAD}}", final_payload)
    


def main():
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file, log_level=log_config.log_level)
    logger.info(f"xboa sdk config loaded...")

    OCR_DIR = Path('~/code/github/boxgpt/docs/ocr').expanduser()
    SCHEMA_DIR = Path('~/code/github/boxgpt/scripts/schema/ab').expanduser()
    
    # variables needed to create the prompt 
    smp_text_path = OCR_DIR / 'ab_smp_tpp_om_014.json' 
    smp_schema_path = SCHEMA_DIR / 'smp-schema.json' 
    user_content_path = SCHEMA_DIR / 'user_prompt.txt'
    system_prompt_path = SCHEMA_DIR / 'system_prompt.txt'


    try:
        
        payload:QueryPayload = QueryPayload(
            smp_schema_path=smp_schema_path,
            smp_text_path=smp_text_path,
            user_content_path=user_content_path,
            system_prompt_path=system_prompt_path)
        
        api_keys = AppConfig.get("api_keys")
        sonnet_api_key = api_keys["DO_SONNET_055"]

        client = OpenAI(base_url="https://inference.do-ai.run/v1/", api_key=sonnet_api_key, timeout=60.0)
        response = client.chat.completions.create(
            model="anthropic-claude-sonnet-5.5",
            messages=[
                {"role": "system", "content": payload.get_system_prompt()},
                {"role": "user", "content": payload.get_user_content()}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        extracted_json = json.loads(response.choices[0].message.content)
        print("Extraction Successful with Target Schema:")
        print(json.dumps(extracted_json, indent=2))
        
    except Exception as e:
            print(f"Extraction failed: {e}")


if __name__ == "__main__":
     main()