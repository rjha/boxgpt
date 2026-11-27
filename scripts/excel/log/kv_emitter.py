from __future__ import annotations
import json
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from scripts.constants import XOUT_DIR


class FlatLogRules(BaseModel):
    skip_hidden_rows: bool = True
    skip_hidden_cells: bool = True
    ignore_empty_columns: bool = True
    date_output_format: str = "%Y-%m-%d"
    drop_summary_rows: bool = True
    min_populated_ratio: float = 0.25
    required_non_empty_cols: List[str] = Field(
        default_factory=lambda: ["Posting Date", "Resource"]
    )
    error_prefixes: List[str] = Field(default_factory=lambda: ["#"])


class FlatLogPolicy(BaseModel):
    policy_version: str = "1.0"
    sheet_shape: str = "flat_log"
    rules: FlatLogRules = Field(default_factory=FlatLogRules)


class LogRecord(BaseModel):
    record_id: str
    row_idx: int
    fields: Dict[str, Any]


class CellErrorRecord(BaseModel):
    row_idx: int
    coord: Optional[str] = None
    col_key: str
    error_token: str


class FlatLogEmitterOutput(BaseModel):
    sheet_name: Optional[str] = None
    records: List[LogRecord] = Field(default_factory=list)
    ignored_columns: List[str] = Field(default_factory=list)
    dropped_summary_rows: List[Dict[str, Any]] = Field(default_factory=list)
    cell_errors: List[CellErrorRecord] = Field(default_factory=list)


def is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        if value.strip() == "":
            return False
    return True


def is_error_token(value: Any, prefixes: List[str]) -> bool:
    if isinstance(value, str):
        cleaned = value.strip()
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                return True
    return False


def normalize_cell_value(value: Any, date_fmt: str) -> Any:
    if isinstance(value, datetime):
        return value.strftime(date_fmt)
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    if isinstance(value, str):
        return value.strip()
    return value


def find_header_row(
    rows: List[Dict[str, Any]],
    skip_hidden: bool,
) -> tuple[Optional[int], Dict[str, str]]:
    for row in rows:
        if skip_hidden and row.get("hidden_row", False):
            continue

        col_headers: Dict[str, str] = {}
        for cell in row.get("cells", []):
            if skip_hidden and cell.get("hidden_cell", False):
                continue
            val = cell.get("value")
            if isinstance(val, str) and val.strip():
                col_headers[cell["col_letter"]] = val.strip()

        # A flat log header row typically names at least 5 distinct columns
        if len(col_headers) >= 5:
            return row["row_idx"], col_headers

    return None, {}


def emit_flat_log_records(
    raw_dump: Dict[str, Any],
    policy: FlatLogPolicy,
) -> FlatLogEmitterOutput:
    rows = sorted(raw_dump.get("rows", []), key=lambda r: r["row_idx"])
    header_row_idx, headers_by_col = find_header_row(
        rows,
        skip_hidden=policy.rules.skip_hidden_rows,
    )

    if header_row_idx is None:
        return FlatLogEmitterOutput(sheet_name=raw_dump.get("sheet_name"))

    candidate_cols = list(headers_by_col.keys())
    populated_cols = set()

    for row in rows:
        if row["row_idx"] <= header_row_idx:
            continue
        if policy.rules.skip_hidden_rows and row.get("hidden_row", False):
            continue
        for cell in row.get("cells", []):
            col_l = cell["col_letter"]
            if col_l not in headers_by_col:
                continue
            if policy.rules.skip_hidden_cells and cell.get("hidden_cell", False):
                continue
            if is_non_empty(cell.get("value")):
                populated_cols.add(col_l)

    if policy.rules.ignore_empty_columns:
        ignored_columns = [c for c in candidate_cols if c not in populated_cols]
        active_cols = [c for c in candidate_cols if c in populated_cols]
    else:
        ignored_columns = []
        active_cols = candidate_cols

    required_cols = [
        c for c in active_cols
        if headers_by_col[c] in policy.rules.required_non_empty_cols
    ]

    output = FlatLogEmitterOutput(
        sheet_name=raw_dump.get("sheet_name"),
        ignored_columns=ignored_columns,
    )

    for row in rows:
        row_idx = row["row_idx"]
        if row_idx <= header_row_idx:
            continue
        if policy.rules.skip_hidden_rows and row.get("hidden_row", False):
            continue

        cells_by_col = {c["col_letter"]: c for c in row.get("cells", [])}
        non_empty_count = 0
        row_values: Dict[str, Any] = {}

        for col_l in active_cols:
            cell = cells_by_col.get(col_l, {})
            if policy.rules.skip_hidden_cells and cell.get("hidden_cell", False):
                row_values[col_l] = None
                continue
            val = cell.get("value")
            if is_non_empty(val):
                non_empty_count += 1
            row_values[col_l] = val

        if non_empty_count == 0:
            continue

        populated_ratio = (non_empty_count / len(active_cols)) if active_cols else 0.0
        missing_required = any(
            not is_non_empty(row_values.get(req_c))
            for req_c in required_cols
        )

        if (
            policy.rules.drop_summary_rows
            and populated_ratio < policy.rules.min_populated_ratio
            and missing_required
        ):
            output.dropped_summary_rows.append({
                "row_idx": row_idx,
                "non_empty_count": non_empty_count,
                "populated_ratio": round(populated_ratio, 3),
            })
            continue

        clean_fields: Dict[str, Any] = {}
        for col_l in active_cols:
            col_key = headers_by_col[col_l]
            raw_val = row_values[col_l]
            cell = cells_by_col.get(col_l, {})

            if not is_non_empty(raw_val):
                clean_fields[col_key] = None
            elif is_error_token(raw_val, policy.rules.error_prefixes):
                output.cell_errors.append(
                    CellErrorRecord(
                        row_idx=row_idx,
                        coord=cell.get("coord"),
                        col_key=col_key,
                        error_token=str(raw_val).strip(),
                    )
                )
                clean_fields[col_key] = None
            else:
                clean_fields[col_key] = normalize_cell_value(
                    raw_val,
                    policy.rules.date_output_format,
                )

        output.records.append(
            LogRecord(
                record_id=f"row_{row_idx}",
                row_idx=row_idx,
                fields=clean_fields,
            )
        )

    return output


def process_sheet(
    sheet_file: Path,
    policy_data: str,
) -> FlatLogEmitterOutput:
    with open(sheet_file, "r", encoding="utf-8") as f:
        raw_dump = json.load(f)
    policy = FlatLogPolicy.model_validate(policy_data)
    return emit_flat_log_records(raw_dump, policy)


def run_process(
    source_dir: Path, 
    target_dir: Path, 
    policy_file: Path
) -> list[Path]:
    with open(policy_file, "r", encoding="utf-8") as f:
        policy_data = json.load(f)

        
    outputs: list[Path] = []
    sheet_files = sorted(source_dir.glob("*.json"))
    for sheet_file in sheet_files:
        emitter_output = process_sheet(sheet_file=sheet_file, policy_data=policy_data)
        emitter_json: str = emitter_output.model_dump_json(indent=2)
        out_path = target_dir / f"{sheet_file.stem}.kv.json"
        out_path.write_text(emitter_json, encoding="utf-8")
        outputs.append(out_path)

    return outputs



def main():
    source_dir = XOUT_DIR / "excel" / "sheet"
    target_dir = XOUT_DIR / "excel" / "kv"
    policy_file = Path("./scripts/excel/policy/flat_log.json")

    run_process(
        source_dir=source_dir, 
        target_dir=target_dir, 
        policy_file=policy_file
    )


if __name__ == "__main__":
    main()
 

