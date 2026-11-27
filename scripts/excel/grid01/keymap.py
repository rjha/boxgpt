import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
from scripts.excel.grid01.model import SheetKeyMap
from scripts.constants import DOC_DIR, XOUT_DIR
from softmaxx.config import AppConfig, get_logger_config
from softmaxx.boxgpt.client.llm import DigitalOceanClient

logger = logging.getLogger("main." + __name__)
# get a lazy instance 
_client: DigitalOceanClient | None = None
ROW_PREFIX_RE = re.compile(r"^-\s+\*\*Row\s+(\d+)")


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
    verbose: bool = False


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
    """Extract and parse a JSON object from raw LLM output, with or without fences."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : idx + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("Top-level JSON value must be an object.")
                return parsed

    raise ValueError("Unterminated JSON object in LLM response.")


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
        match = ROW_PREFIX_RE.match(line.strip())
        if not match:
            continue
        r_num = int(match.group(1))
        if r_num not in mapped_rows and r_num not in ignored_rows:
            warnings.append(
                f"Row {r_num} was neither mapped in row_keys nor listed in ignore_rows."
            )

    return warnings


def call_llm_for_keymap(skeleton_text: str, config: Stage3Config) -> str:
    
    system_prompt = config.prompt_path.read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": skeleton_text},
    ]

    response = _get_client().chat_stream(
        model=config.model,
        llm_messages=messages,
        temperature=config.temperature,
        verbose=config.verbose,
    )
    return _extract_raw_text(response)



def run_skeleton_to_keymap(
    skeleton_dir: Path,
    target_dir: Path,
    config: Stage3Config,
) -> list[Path]:
    """Run Stage 3 over *.skeleton.md -> *.keymap.json (plus *.warnings.json if needed)."""
    # src = Path(skeleton_dir)
    # dst = Path(target_dir)

    outputs: list[Path] = []
    skeleton_files = sorted(skeleton_dir.glob("*.skeleton.md"))
    logger.info(f"Found {len(skeleton_files)} skeleton files in dir {skeleton_dir}")

    for skeleton_file in skeleton_files:
        logger.info(f"=> processing skeleton file {skeleton_file}")
        skeleton_text = skeleton_file.read_text(encoding="utf-8")
        stem = skeleton_file.name.removesuffix(".skeleton.md")
        llm_response_file = target_dir / f"{stem}.raw.txt"

        # store LLM response text on disk 
        # do not call LLM if this file exists 
        if llm_response_file.exists() and llm_response_file.stat().st_size > 0:
            logger.info(f"loading LLM response from {llm_response_file}")
            raw_text = llm_response_file.read_text(encoding="utf-8")
        else:
            logger.info("call LLM to get key mappings")
            raw_text = call_llm_for_keymap(skeleton_text, config)
            llm_response_file.write_text(raw_text, encoding="utf-8")

        logger.info("parse LLM key mapping response")
        payload = parse_json_from_llm_text(raw_text)
        # validate LLM response
        logger.info("validate LLM response")
        key_map = SheetKeyMap.model_validate(payload)

        warnings = validate_key_map(key_map, skeleton_text)
        # stem = skeleton_file.name.removesuffix(".skeleton.md")
        map_out = target_dir / f"{stem}.keymap.json"
        map_out.write_text(key_map.model_dump_json(indent=2), encoding="utf-8")
        outputs.append(map_out)

        if warnings:
            warn_out = target_dir / f"{stem}.warnings.json"
            warn_out.write_text(json.dumps(warnings, indent=2), encoding="utf-8")
        
        logger.info("skeleton file processed")
        time.sleep(2)

    return outputs


def main():
    source_dir = XOUT_DIR / "excel" / "skeleton"
    target_dir = XOUT_DIR / "excel" / "keymap"
    resource_dir = Path(AppConfig.get("resource_dir")).resolve()
    keymap_prompt_dir = resource_dir / "prompts" / "pyxl" / "keymap"
    keymap_prompt_path = keymap_prompt_dir / "prompt.txt"
    
    config: Stage3Config = Stage3Config(
        model=DigitalOceanClient.SONNET_55_MODEL,
        prompt_path=keymap_prompt_path,
        verbose=True
    )

    run_skeleton_to_keymap(
        skeleton_dir=source_dir, 
        target_dir=target_dir, 
        config=config
    )

if __name__ == "__main__":
    AppConfig.load()
    local_log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=local_log_config.log_file)
    main()
