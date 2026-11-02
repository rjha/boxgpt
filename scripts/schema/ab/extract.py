import json
import logging
from pathlib import Path
from openai import OpenAI
from softmaxx.config import AppConfig, get_logger_config


logger = logging.getLogger("main." + __name__)


def read_text_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.read_text(encoding="utf-8")




def main():
    AppConfig.load()
    log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=log_config.log_file, log_level=log_config.log_level)
    logger.info(f"xboa sdk config loaded...")

    # variables needed to create the prompt 
    smp_text_path = Path('~/code/github/boxgpt/docs/ocr/ab_smp_tpp_om_014.json').expanduser()
    smp_schema_path = Path('~/code/github/boxgpt/scripts/schema/ab/smp-schema.json').expanduser()
    user_content_template_path = Path('~/code/github/boxgpt/scripts/schema/ab/user_prompt.txt').expanduser()
    system_prompt_path = Path('~/code/github/boxgpt/scripts/schema/ab/system_prompt.txt').expanduser()

    sample_metadata = {"header_summary": "Title: Battery bank | Doc: SMP/E&I/005"}
    sample_ppe = [{"activity": "Battery terminal cleaning", "required_ppe": ["Helmet", "Goggles"]}]

    try:
        
        llm_payload = {
            "metadata_hints": sample_metadata,
            "ppe_matrices": sample_ppe,
            "document_text": read_text_file(smp_text_path)
        }

        final_payload = json.dumps(llm_payload, indent=2)

        user_content_template = read_text_file(user_content_template_path)
        smp_schema = read_text_file(smp_schema_path)
        system_prompt = read_text_file(system_prompt_path)

        # replace content into user prompts
        user_content = user_content_template.replace("{{SCHEMA}}", smp_schema).replace("{{PAYLOAD}}", final_payload)
        api_keys = AppConfig.get("api_keys")
        sonnet_api_key = api_keys["DO_SONNET_055"]

        client = OpenAI(base_url="https://inference.do-ai.run/v1/", api_key=sonnet_api_key, timeout=60.0)
        response = client.chat.completions.create(
            model="anthropic-claude-sonnet-5.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
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