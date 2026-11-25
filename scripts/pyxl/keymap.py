import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
from scripts.pyxl.model import SheetKeyMap
from scripts.constants import DOC_DIR, XOUT_DIR
from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.client.llm import DigitalOceanClient

logger = logging.getLogger("main." + __name__)

# get a lazy instance 
_client: DigitalOceanClient | None = None

def _get_client() -> DigitalOceanClient:
    """Lazily create DigitalOceanClient once after AppConfig is initialized."""
    global _client
    if _client is None:
       api_keys = AppConfig.get("api_keys")
       _client = DigitalOceanClient(api_key=api_keys["DO_INFERENCE_KEY"])

    return _client

@dataclass(frozen=True)
class Stage3Config:
    """Configuration for Stage 3: skeleton -> key map via DigitalOceanClient.chat_02."""

    model: str
    prompt_path: Path
    temperature: float = 0.0
    verbose: bool = True


def _extract_raw_text(response: Any) -> str:
    """Extract text payload from DigitalOceanClient.chat_02 response."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("content", "text", "message", "output"):
            val = response.get(key)
            if isinstance(val, str) and val.strip():
                return val
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    for attr in ("content", "text", "output_text"):
        val = getattr(response, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    choices = getattr(response, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
    raise TypeError(f"Unsupported response type from chat_02: {type(response).__name__}")


def parse_json_from_llm_text(raw_text: str) -> dict[str, Any]:
    """Strip optional markdown fences and parse raw JSON text."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def validate_key_map(key_map: SheetKeyMap, skeleton_text: str) -> list[str]:
    """Check for duplicate target row keys and unmapped skeleton rows."""
    warnings: list[str] = []

    if not key_map.col_keys:
        warnings.append("col_keys is empty.")
    if not key_map.row_keys:
        warnings.append("row_keys is empty.")

    seen_row_keys: dict[str, str] = {}
    for row_idx, r_key in key_map.row_keys.items():
        if r_key in seen_row_keys:
            warnings.append(
                f"Duplicate row key '{r_key}' on rows {seen_row_keys[r_key]} and {row_idx}."
            )
        else:
            seen_row_keys[r_key] = row_idx

    mapped_rows = {int(r) for r in key_map.row_keys.keys()}
    ignored_rows = set(key_map.ignore_rows)
    for line in skeleton_text.splitlines():
        if line.startswith("- **Row "):
            row_str = line.split("**Row ")[1].split("**")[0].split()[0]
            r_num = int(row_str)
            if r_num not in mapped_rows and r_num not in ignored_rows:
                warnings.append(f"Row {r_num} was neither mapped in row_keys nor listed in ignore_rows.")

    return warnings


def call_llm_for_keymap(skeleton_text: str, config: Stage3Config) -> str:
    
    system_prompt = config.prompt_path.read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": skeleton_text},
    ]

   
    response = _get_client().chat_02(
        model=config.model,
        llm_messages=messages,
        temperature=config.temperature,
        verbose=config.verbose,
    )
    return _extract_raw_text(response)



def run_stage3(
    skeleton_dir: str | Path,
    target_dir: str | Path,
    config: Stage3Config,
) -> list[Path]:
    """Run Stage 3 over *.skeleton.md -> *.keymap.json (plus *.warnings.json if needed)."""
    src = Path(skeleton_dir)
    dst = Path(target_dir)

    outputs: list[Path] = []
    for skel_path in sorted(src.glob("*.skeleton.md")):
        skeleton_text = skel_path.read_text(encoding="utf-8")
        # get LLM response 
        raw_text = call_llm_for_keymap(skeleton_text, config)
        payload = parse_json_from_llm_text(raw_text)
        # validate LLM response
        key_map = SheetKeyMap.model_validate(payload)

        warnings = validate_key_map(key_map, skeleton_text)
        stem = skel_path.name.removesuffix(".skeleton.md")
        map_out = dst / f"{stem}.keymap.json"
        map_out.write_text(key_map.model_dump_json(indent=2), encoding="utf-8")
        outputs.append(map_out)

        if warnings:
            warn_out = dst / f"{stem}.warnings.json"
            warn_out.write_text(json.dumps(warnings, indent=2), encoding="utf-8")

    return outputs

def main():
    source_dir = DOC_DIR / "excel" / "stage02"
    target_dir = XOUT_DIR / "excel" / "stage03"
    resource_dir = Path(AppConfig.get("resource_dir")).resolve()
    keymap_prompt_path = resource_dir / "pyxl" / "keymap" / "prompt.txt"
    
    config: Stage3Config = Stage3Config(
        model=DigitalOceanClient.SONNET_55_MODEL,
        prompt_path=keymap_prompt_path
    )

    run_stage3(
        skeleton_dir=source_dir, 
        target_dir=target_dir, 
        config=config
    )

if __name__ == "__main__":
    AppConfig.load()
    local_log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=local_log_config.log_file)
    main()
