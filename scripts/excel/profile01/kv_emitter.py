import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from scripts.constants import DOC_DIR, XOUT_DIR
from softmaxx.config import AppConfig, get_logger_config


logger = logging.getLogger("main." + __name__)
DUPLICATE_ROW_KEY_RE = re.compile(r"^Duplicate row key '.*?' on rows (.*?)\.$")
UNMAPPED_ROW_RE = re.compile(
    r"^Row (\d+) was neither mapped in row_keys nor listed in ignore_rows\.$"
)


@dataclass(frozen=True)
class Stage4Config:
    """Configuration for Stage 4: build JSON KV store from Stage 1 + Stage 3 outputs."""

    include_nulls: bool = False
    key_separator: str = "::"


def _parse_warning_duplicate_rows(warnings: list[str]) -> set[int]:
    """Extract row numbers mentioned in duplicate row-key warnings."""
    flagged_rows: set[int] = set()
    for msg in warnings:
        match = DUPLICATE_ROW_KEY_RE.match(msg.strip())
        if not match:
            continue
        for token in re.findall(r"\d+", match.group(1)):
            flagged_rows.add(int(token))
    return flagged_rows


def _parse_warning_unmapped_rows(warnings: list[str]) -> set[int]:
    """Extract row numbers mentioned in unmapped-row warnings."""
    unmapped_rows: set[int] = set()
    for msg in warnings:
        match = UNMAPPED_ROW_RE.match(msg.strip())
        if match:
            unmapped_rows.add(int(match.group(1)))
    return unmapped_rows


def _unwrap_cell_value(cell: Any) -> Any:
    """Extract scalar value from a Stage 1 cell object."""
    if isinstance(cell, dict):
        return cell.get("value")
    return cell


def _get_sheet_container(stage1_data: dict[str, Any], sheet_title: str) -> dict[str, Any]:
    """Locate the sheet container inside a Stage 1 JSON payload."""
    if isinstance(stage1_data.get("sheets"), dict):
        container = stage1_data["sheets"].get(sheet_title, {})
    elif isinstance(stage1_data.get(sheet_title), dict):
        container = stage1_data[sheet_title]
    else:
        container = stage1_data

    return container if isinstance(container, dict) else {}


def _extract_hidden_rows(stage1_data: dict[str, Any], sheet_title: str) -> set[int]:
    """Collect row indices marked with hidden_row=True in Stage 1 JSON."""
    container = _get_sheet_container(stage1_data, sheet_title)
    raw_rows = container.get("rows")
    hidden: set[int] = set()

    if isinstance(raw_rows, list):
        for row_obj in raw_rows:
            if isinstance(row_obj, dict) and row_obj.get("hidden_row") is True:
                r_raw = row_obj.get("row_idx", row_obj.get("id", row_obj.get("row")))
                if r_raw is not None and str(r_raw).isdigit():
                    hidden.add(int(r_raw))

    return hidden


def _extract_sheet_cells(stage1_data: dict[str, Any], sheet_title: str) -> dict[int, dict[str, Any]]:
    """
    Normalize Stage 1 sheet data into {row_num: {col_letter: cell_value}}.
    Supports `rows` as a list of {"row_idx": int, "cells": [{"col_letter": str, "value": ...}]}
    or dict-based layouts as a fallback.
    """
    container = _get_sheet_container(stage1_data, sheet_title)
    raw_rows = container.get("rows")
    normalized: dict[int, dict[str, Any]] = {}

    if isinstance(raw_rows, list):
        for row_obj in raw_rows:
            if not isinstance(row_obj, dict):
                continue
            r_raw = row_obj.get("row_idx", row_obj.get("id", row_obj.get("row")))
            if r_raw is None or not str(r_raw).isdigit():
                continue
            r_num = int(r_raw)
            raw_cells = row_obj.get("cells", [])

            row_map: dict[str, Any] = {}
            if isinstance(raw_cells, list):
                for cell_obj in raw_cells:
                    if not isinstance(cell_obj, dict):
                        continue
                    col_letter = cell_obj.get("col_letter") or cell_obj.get("col")
                    if col_letter:
                        row_map[str(col_letter)] = _unwrap_cell_value(cell_obj)
            elif isinstance(raw_cells, dict):
                for col_k, cell_v in raw_cells.items():
                    row_map[str(col_k)] = _unwrap_cell_value(cell_v)

            normalized[r_num] = row_map
        return normalized

    if isinstance(raw_rows, dict):
        for row_k, row_val in raw_rows.items():
            if not str(row_k).isdigit() or not isinstance(row_val, dict):
                continue
            normalized[int(row_k)] = {
                str(col_k): _unwrap_cell_value(cell_v) for col_k, cell_v in row_val.items()
            }
        return normalized

    raise ValueError(f"Could not locate row/cell data for sheet '{sheet_title}' in Stage 1 JSON.")


def resolve_active_rows(
    row_keys: dict[str, str],
    hidden_rows: list[int] | set[int],
    ignore_rows: list[int] | set[int],
    warnings: list[str],
) -> tuple[dict[int, str], dict[str, list[int]]]:
    """
    Return:
      - active_rows: {row_num: row_key} to emit into the KV store
      - skipped_conflicts: {row_key: sorted_row_nums} skipped due to collision
    """
    hidden_set = set(hidden_rows)
    ignore_set = set(ignore_rows)
    warning_rows = _parse_warning_duplicate_rows(warnings)

    rows_by_key: dict[str, list[int]] = defaultdict(list)
    for r_str, r_key in row_keys.items():
        r_num = int(r_str)
        if r_num in ignore_set:
            continue
        rows_by_key[r_key].append(r_num)

    active_rows: dict[int, str] = {}
    skipped_conflicts: dict[str, list[int]] = {}

    for r_key, r_nums in rows_by_key.items():
        ordered = sorted(r_nums)
        if len(ordered) == 1:
            r_num = ordered[0]
            if r_num not in hidden_set:
                active_rows[r_num] = r_key
            continue

        has_hidden = any(r in hidden_set for r in ordered)
        is_flagged_scratch = any(r in warning_rows for r in ordered)

        if has_hidden and not is_flagged_scratch:
            visible_candidates = [r for r in ordered if r not in hidden_set]
            winner = max(visible_candidates) if visible_candidates else max(ordered)
            active_rows[winner] = r_key
        else:
            skipped_conflicts[r_key] = ordered

    return active_rows, skipped_conflicts


def build_sheet_kv(
    sheet_data: dict[str, Any],
    keymap_data: dict[str, Any],
    warnings: list[str],
    config: Stage4Config,
) -> dict[str, Any]:
    """Build flat KV dictionary for a single sheet."""
    sheet_title = str(keymap_data.get("sheet_title", ""))
    col_keys: dict[str, str] = keymap_data.get("col_keys", {})
    row_keys: dict[str, str] = keymap_data.get("row_keys", {})
    keymap_hidden_rows: list[int] = keymap_data.get("hidden_rows", [])
    ignore_rows: list[int] = keymap_data.get("ignore_rows", [])

    stage1_hidden_rows = _extract_hidden_rows(sheet_data, sheet_title)
    merged_hidden_rows = sorted(set(keymap_hidden_rows) | stage1_hidden_rows)

    active_rows, skipped_conflicts = resolve_active_rows(
        row_keys=row_keys,
        hidden_rows=merged_hidden_rows,
        ignore_rows=ignore_rows,
        warnings=warnings,
    )
    unmapped_rows = sorted(_parse_warning_unmapped_rows(warnings))

    if skipped_conflicts:
        logger.warning(
            "Sheet '%s': skipping %d conflicted row_keys: %s",
            sheet_title,
            len(skipped_conflicts),
            skipped_conflicts,
        )

    sheet_cells = _extract_sheet_cells(sheet_data, sheet_title)
    kv_store: dict[str, Any] = {}

    for r_num in sorted(active_rows.keys()):
        r_key = active_rows[r_num]
        row_cells = sheet_cells.get(r_num, {})
        for col_letter, c_key in col_keys.items():
            if col_letter not in row_cells:
                continue
            val = row_cells[col_letter]
            if val is None and not config.include_nulls:
                continue
            kv_key = f"{r_key}{config.key_separator}{c_key}"
            kv_store[kv_key] = val

    return {
        "sheet_title": sheet_title,
        "resolved_rows": {str(k): v for k, v in sorted(active_rows.items())},
        "skipped_conflicts": skipped_conflicts,
        "unmapped_warning_rows": unmapped_rows,
        "kv": kv_store,
    }


def run_stage4(
    sheet_dir: Path,
    keymap_dir: Path,
    target_dir: Path,
    config: Stage4Config,
) -> list[Path]:
    """Run [keymap/*.keymap.json + sheet/*.json]  -> kv/*.kv.json."""
    
    outputs: list[Path] = []
    keymap_files = sorted(keymap_dir.glob("*.keymap.json"))

    for keymap_file in keymap_files:
        logger.info(f"process keymap file: {keymap_file}")
        stem = keymap_file.name.removesuffix(".keymap.json")
        warning_file = keymap_dir / f"{stem}.warnings.json"
        sheet_file = sheet_dir / f"{stem}.json"

        keymap_data = json.loads(keymap_file.read_text(encoding="utf-8"))
        warnings = (
            json.loads(warning_file.read_text(encoding="utf-8"))
            if warning_file.exists()
            else []
        )

        sheet_data = json.loads(sheet_file.read_text(encoding="utf-8"))
        result = build_sheet_kv(
            sheet_data=sheet_data,
            keymap_data=keymap_data,
            warnings=warnings,
            config=config,
        )

        out_file = target_dir / f"{stem}.kv.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        outputs.append(out_file)
        logger.info("Wrote %s (%d KV entries)", out_file.name, len(result["kv"]))

    return outputs


def main() -> None:
    sheet_dir = XOUT_DIR / "excel" / "sheet"
    keymap_dir = XOUT_DIR / "excel" / "keymap"
    target_dir = XOUT_DIR / "excel" / "kv"

    config = Stage4Config(include_nulls=False, key_separator="::")
    run_stage4(
        sheet_dir=sheet_dir,
        keymap_dir=keymap_dir,
        target_dir=target_dir,
        config=config,
    )


if __name__ == "__main__":
    AppConfig.load()
    local_log_config = get_logger_config("local")
    AppConfig.init_logging(log_file=local_log_config.log_file)
    main()