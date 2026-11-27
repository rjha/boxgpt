
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from openpyxl.utils import column_index_from_string
from pydantic import BaseModel, Field
from scripts.constants import XOUT_DIR


class SkipHiddenRule(BaseModel):
    enabled: bool = True
    skip_hidden_rows: bool = True
    skip_hidden_cells: bool = True
    record_skipped: bool = True


class FullSpanMergedRule(BaseModel):
    enabled: bool = True
    min_width_coverage: float = 1.0
    treat_as: str = "non_data_banner"


class EmptyColumnsRule(BaseModel):
    enabled: bool = True
    record_ignored: bool = True


class ColumnHeadersRule(BaseModel):
    auto_detect_header_rows: bool = True
    date_output_format: str = "%b-%y"
    uppercase_dates: bool = True
    join_separator: str = " / "


class SectionHierarchyRule(BaseModel):
    lookahead_cells: int = 20
    max_occupancy_ratio: float = 0.20
    key_separator: str = " / "
    record_empty_header_keys: bool = True


class UnlabeledRowsRule(BaseModel):
    ignore_with_data: bool = True
    record_ignored: bool = True


class ErrorCellsRule(BaseModel):
    enabled: bool = True
    error_prefixes: List[str] = Field(default_factory=lambda: ["#"])
    record_errors: bool = True


class PolicyRules(BaseModel):
    skip_hidden: SkipHiddenRule = Field(default_factory=SkipHiddenRule)
    full_span_merged_rows: FullSpanMergedRule = Field(default_factory=FullSpanMergedRule)
    empty_columns: EmptyColumnsRule = Field(default_factory=EmptyColumnsRule)
    column_headers: ColumnHeadersRule = Field(default_factory=ColumnHeadersRule)
    section_hierarchy: SectionHierarchyRule = Field(default_factory=SectionHierarchyRule)
    unlabeled_rows: UnlabeledRowsRule = Field(default_factory=UnlabeledRowsRule)
    error_cells: ErrorCellsRule = Field(default_factory=ErrorCellsRule)


class KvEmitterPolicy(BaseModel):
    policy_version: str = "2.0"
    rules: PolicyRules = Field(default_factory=PolicyRules)


class KvRecord(BaseModel):
    row_idx: int
    coord: Optional[str] = None
    row_key: str
    col_key: str
    value: Any


class CellErrorRecord(BaseModel):
    row_idx: int
    coord: Optional[str] = None
    row_key: str
    col_key: str
    error_token: str


class EmitterOutput(BaseModel):
    sheet_name: Optional[str] = None
    kv_records: List[KvRecord] = Field(default_factory=list)
    hidden_items: List[Dict[str, Any]] = Field(default_factory=list)
    ignored_columns: List[str] = Field(default_factory=list)
    ignored_rows: List[Dict[str, Any]] = Field(default_factory=list)
    cell_errors: List[CellErrorRecord] = Field(default_factory=list)
    empty_section_keys: List[Dict[str, Any]] = Field(default_factory=list)


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


def format_header_token(value: Any, date_fmt: str, uppercase_dates: bool) -> Optional[str]:
    if not is_non_empty(value):
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
        formatted = parsed.strftime(date_fmt)
        if uppercase_dates:
            return formatted.upper()
        return formatted
    except ValueError:
        return text


def is_full_span_merged_row(
    row_obj: Dict[str, Any],
    total_cols: int,
    min_coverage: float,
) -> bool:
    if total_cols <= 0:
        return False
    for cell in row_obj.get("cells", []):
        merge_info = cell.get("merge")
        if isinstance(merge_info, dict) and merge_info.get("range"):
            rng = str(merge_info["range"])
            if ":" in rng:
                start_ref, end_ref = rng.split(":", 1)
                start_col = "".join(ch for ch in start_ref if ch.isalpha())
                end_col = "".join(ch for ch in end_ref if ch.isalpha())
                if start_col and end_col:
                    span = column_index_from_string(end_col) - column_index_from_string(start_col) + 1
                    if (span / total_cols) >= min_coverage:
                        return True
    return False


def detect_ignored_columns(
    rows: List[Dict[str, Any]],
    all_cols: List[str],
    policy: KvEmitterPolicy,
) -> List[str]:
    if not policy.rules.empty_columns.enabled:
        return []

    populated_cols = set()
    for row in rows:
        if policy.rules.skip_hidden.enabled and policy.rules.skip_hidden.skip_hidden_rows:
            if row.get("hidden_row", False):
                continue
        for cell in row.get("cells", []):
            if policy.rules.skip_hidden.enabled and policy.rules.skip_hidden.skip_hidden_cells:
                if cell.get("hidden_cell", False):
                    continue
            if is_non_empty(cell.get("value")):
                populated_cols.add(cell["col_letter"])

    ignored: List[str] = []
    for col_letter in all_cols:
        if col_letter not in populated_cols:
            ignored.append(col_letter)
    return ignored


def detect_first_metric_row(
    rows: List[Dict[str, Any]],
    active_cols: List[str],
    total_cols: int,
    policy: KvEmitterPolicy,
) -> Optional[int]:
    for row in rows:
        if policy.rules.skip_hidden.enabled and policy.rules.skip_hidden.skip_hidden_rows:
            if row.get("hidden_row", False):
                continue
        if policy.rules.full_span_merged_rows.enabled:
            if is_full_span_merged_row(row, total_cols, policy.rules.full_span_merged_rows.min_width_coverage):
                continue

        numeric_hits = 0
        for cell in row.get("cells", []):
            if cell["col_letter"] in active_cols:
                val = cell.get("value")
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    numeric_hits += 1

        if numeric_hits >= 3:
            return row["row_idx"]
    return None


def detect_label_column(
    rows: List[Dict[str, Any]],
    candidate_cols: List[str],
    after_row_idx: int,
    policy: KvEmitterPolicy,
) -> Optional[str]:
    best_col: Optional[str] = None
    best_score = -1

    for col_letter in candidate_cols:
        text_count = 0
        for row in rows:
            if row["row_idx"] <= after_row_idx:
                continue
            if policy.rules.skip_hidden.enabled and policy.rules.skip_hidden.skip_hidden_rows:
                if row.get("hidden_row", False):
                    continue
            cell = next((c for c in row.get("cells", []) if c["col_letter"] == col_letter), None)
            if cell and isinstance(cell.get("value"), str) and cell["value"].strip():
                text_count += 1

        if text_count > best_score:
            best_score = text_count
            best_col = col_letter

    return best_col


def build_column_key_map(
    header_rows: List[Dict[str, Any]],
    data_cols: List[str],
    policy: KvEmitterPolicy,
) -> Dict[str, str]:
    col_map: Dict[str, str] = {}
    sep = policy.rules.column_headers.join_separator
    date_fmt = policy.rules.column_headers.date_output_format
    upper_dates = policy.rules.column_headers.uppercase_dates

    for col_letter in data_cols:
        tokens: List[str] = []
        for h_row in header_rows:
            cell = next((c for c in h_row.get("cells", []) if c["col_letter"] == col_letter), None)
            if cell:
                tok = format_header_token(cell.get("value"), date_fmt, upper_dates)
                if tok and tok not in tokens:
                    tokens.append(tok)
        if tokens:
            col_map[col_letter] = sep.join(tokens)
        else:
            col_map[col_letter] = col_letter

    return col_map


def emit_kv_records(raw_dump: Dict[str, Any], policy: KvEmitterPolicy) -> EmitterOutput:
    rows = sorted(raw_dump.get("rows", []), key=lambda r: r["row_idx"])
    total_cols = len(rows[0]["cells"]) if rows else 0
    all_col_letters = [c["col_letter"] for c in rows[0]["cells"]] if rows else []

    ignored_columns = detect_ignored_columns(rows, all_col_letters, policy)
    non_empty_cols = [c for c in all_col_letters if c not in ignored_columns]

    first_metric_row = detect_first_metric_row(rows, non_empty_cols, total_cols, policy)

    header_row_objs: List[Dict[str, Any]] = []
    if first_metric_row is not None and policy.rules.column_headers.auto_detect_header_rows:
        for row in rows:
            if row["row_idx"] >= first_metric_row:
                break
            if policy.rules.skip_hidden.enabled and policy.rules.skip_hidden.skip_hidden_rows:
                if row.get("hidden_row", False):
                    continue
            if policy.rules.full_span_merged_rows.enabled:
                if is_full_span_merged_row(row, total_cols, policy.rules.full_span_merged_rows.min_width_coverage):
                    continue
            has_tokens = any(
                is_non_empty(c.get("value"))
                for c in row.get("cells", [])
                if c["col_letter"] in non_empty_cols[2:]
            )
            if has_tokens:
                header_row_objs.append(row)

    header_max_row = header_row_objs[-1]["row_idx"] if header_row_objs else 0
    label_col = detect_label_column(rows, non_empty_cols[:4], header_max_row, policy)

    if label_col and label_col in non_empty_cols:
        label_pos = non_empty_cols.index(label_col)
        data_cols = non_empty_cols[label_pos + 1 :]
        prior_cols = non_empty_cols[:label_pos]
    else:
        label_pos = 0
        data_cols = non_empty_cols
        prior_cols = []

    col_keys = build_column_key_map(header_row_objs, data_cols, policy)

    output = EmitterOutput(
        sheet_name=raw_dump.get("sheet_name"),
        ignored_columns=ignored_columns,
    )

    section_stack: List[str] = []
    kv_sep = policy.rules.section_hierarchy.key_separator
    lookahead_n = policy.rules.section_hierarchy.lookahead_cells
    max_occ = policy.rules.section_hierarchy.max_occupancy_ratio

    for row in rows:
        row_idx = row["row_idx"]

        # Rule 1: Skip hidden rows
        if policy.rules.skip_hidden.enabled and policy.rules.skip_hidden.skip_hidden_rows:
            if row.get("hidden_row", False):
                if policy.rules.skip_hidden.record_skipped:
                    output.hidden_items.append({"type": "row", "row_idx": row_idx})
                continue

        # Skip header rows after building column keys
        if row_idx <= header_max_row:
            continue

        # Rule 2: Exclude full-span merged rows from data
        if policy.rules.full_span_merged_rows.enabled:
            if is_full_span_merged_row(row, total_cols, policy.rules.full_span_merged_rows.min_width_coverage):
                continue

        cells_by_col = {c["col_letter"]: c for c in row.get("cells", [])}
        raw_label = cells_by_col.get(label_col, {}).get("value") if label_col else None
        label = str(raw_label).strip() if is_non_empty(raw_label) else ""

        # Compute occupancy across the first N non-hidden active data cells
        lookahead_cols = data_cols[:lookahead_n]
        occupied_cells = 0
        for col_letter in lookahead_cols:
            cell_obj = cells_by_col.get(col_letter, {})
            if policy.rules.skip_hidden.enabled and policy.rules.skip_hidden.skip_hidden_cells:
                if cell_obj.get("hidden_cell", False):
                    if policy.rules.skip_hidden.record_skipped:
                        output.hidden_items.append({
                            "type": "cell",
                            "coord": cell_obj.get("coord"),
                            "row_idx": row_idx,
                        })
                    continue
            if is_non_empty(cell_obj.get("value")):
                occupied_cells += 1

        occupancy_ratio = (occupied_cells / len(lookahead_cols)) if lookahead_cols else 0.0

        # Rule 3: Unlabeled row with data -> record in ignored_rows
        if not label:
            if occupied_cells > 0 and policy.rules.unlabeled_rows.ignore_with_data:
                if policy.rules.unlabeled_rows.record_ignored:
                    output.ignored_rows.append({
                        "row_idx": row_idx,
                        "reason": "unlabeled_with_data",
                        "occupied_cells": occupied_cells,
                    })
            continue

        # Rule 4: Section / sub-section header via occupancy < threshold
        if occupancy_ratio < max_occ:
            has_prior_marker = any(
                is_non_empty(cells_by_col.get(pc, {}).get("value"))
                for pc in prior_cols
            )
            if has_prior_marker or not section_stack:
                section_stack = [label]
            elif len(section_stack) == 1:
                section_stack.append(label)
            else:
                section_stack[-1] = label

            if occupied_cells == 0 and policy.rules.section_hierarchy.record_empty_header_keys:
                output.empty_section_keys.append({
                    "row_idx": row_idx,
                    "row_key": kv_sep.join(section_stack),
                })
            continue

        # Rule 5: Emit KV records or record error tokens
        if section_stack:
            row_key = kv_sep.join(section_stack + [label])
        else:
            row_key = label

        for col_letter in data_cols:
            cell_obj = cells_by_col.get(col_letter, {})
            if policy.rules.skip_hidden.enabled and policy.rules.skip_hidden.skip_hidden_cells:
                if cell_obj.get("hidden_cell", False):
                    continue

            val = cell_obj.get("value")
            if not is_non_empty(val):
                continue

            if policy.rules.error_cells.enabled and is_error_token(val, policy.rules.error_cells.error_prefixes):
                if policy.rules.error_cells.record_errors:
                    output.cell_errors.append(
                        CellErrorRecord(
                            row_idx=row_idx,
                            coord=cell_obj.get("coord"),
                            row_key=row_key,
                            col_key=col_keys[col_letter],
                            error_token=str(val).strip(),
                        )
                    )
            else:
                output.kv_records.append(
                    KvRecord(
                        row_idx=row_idx,
                        coord=cell_obj.get("coord"),
                        row_key=row_key,
                        col_key=col_keys[col_letter],
                        value=val,
                    )
                )

    return output


def process_sheet(sheet_file: Path, policy_data: str) -> EmitterOutput:
    with open(sheet_file, "r", encoding="utf-8") as f:
                raw_dump = json.load(f)
    
    policy = KvEmitterPolicy.model_validate(policy_data)
    return emit_kv_records(raw_dump, policy)


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
    policy_file = Path("./scripts/excel/policy/grid02.json")
    run_process(
        source_dir=source_dir, 
        target_dir=target_dir, 
        policy_file=policy_file
    )


if __name__ == "__main__":
    main()
 

