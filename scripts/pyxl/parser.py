from datetime import date, datetime, time, timedelta
import json
import os
import re
import openpyxl
from openpyxl.utils import get_column_letter
from scripts.constants import DOC_DIR, XOUT_DIR


def serialize_val(val):
  if isinstance(val, (datetime, date, time)):
    return val.isoformat()
  if isinstance(val, timedelta):
    return val.total_seconds()
  if isinstance(val, float):
    return round(val, 6)
  if val is not None and not isinstance(val, (str, int, bool)):
    return str(val)
  return val


def sanitize_filename(name):
  return re.sub(r'[\\/*?:"<>|]', "_", name.strip()) or "unnamed_sheet"


def dump_workbook_for_llm(excel_path, output_dir):
  
  # Load twice: once for evaluated values, once for raw Excel formulas
  wb_val = openpyxl.load_workbook(excel_path, data_only=True)
  wb_frm = openpyxl.load_workbook(excel_path, data_only=False)

  manifest = {
      "source_file": os.path.basename(excel_path),
      "sheet_count": len(wb_val.sheetnames),
      "sheets": [],
  }

  for idx, sheet_name in enumerate(wb_val.sheetnames):
    ws_val = wb_val[sheet_name]
    ws_frm = wb_frm[sheet_name]

    # Pre-map merged cells so every cell knows its span and master value
    merge_map = {}
    merged_ranges = []
    for rng in ws_val.merged_cells.ranges:
      merged_ranges.append(str(rng))
      top_left_val = serialize_val(ws_val.cell(rng.min_row, rng.min_col).value)
      for r in range(rng.min_row, rng.max_row + 1):
        for c in range(rng.min_col, rng.max_col + 1):
          merge_map[(r, c)] = {
              "range": str(rng),
              "is_origin": (r == rng.min_row and c == rng.min_col),
              "origin_value": top_left_val,
          }

    # Determine active grid column bounds (all columns with at least one non-empty cell or merge)
    active_col_indices = sorted({
        c
        for r in range(1, ws_val.max_row + 1)
        for c in range(1, ws_val.max_column + 1)
        if ws_val.cell(r, c).value is not None
        or ws_frm.cell(r, c).value is not None
        or ws_val.cell(r, c).comment is not None
        or (r, c) in merge_map
    })
    min_col = active_col_indices[0] if active_col_indices else 1
    max_col = active_col_indices[-1] if active_col_indices else 0
    grid_cols = list(range(min_col, max_col + 1)) if max_col >= min_col else []

    rows_data = []
    md_lines = [f"# Sheet: {sheet_name}\n"]
    if merged_ranges:
      md_lines.append(f"**Merged Ranges:** {', '.join(merged_ranges)}\n")

    for r in range(1, ws_val.max_row + 1):
      is_hidden_row = bool(ws_val.row_dimensions[r].hidden)
      row_cells = []
      md_row_tokens = []

      has_row_content = False
      for c in grid_cols:
        col_letter = get_column_letter(c)
        is_hidden_col = bool(ws_val.column_dimensions[col_letter].hidden)

        cell_val = ws_val.cell(r, c)
        cell_frm = ws_frm.cell(r, c)

        val = serialize_val(cell_val.value)
        raw_frm = (
            cell_frm.value
            if isinstance(cell_frm.value, str)
            and cell_frm.value.startswith("=")
            else None
        )
        comment = cell_val.comment.text.strip() if cell_val.comment else None
        merge_info = merge_map.get((r, c))

        if val is not None or raw_frm is not None or comment is not None or merge_info:
          has_row_content = True

        # Structural & hierarchy cues for schema inference
        is_bold = bool(cell_val.font and cell_val.font.bold)
        indent = (
            cell_val.alignment.indent
            if (cell_val.alignment and cell_val.alignment.indent)
            else 0
        )
        num_fmt = (
            cell_val.number_format
            if cell_val.number_format not in (None, "General")
            else None
        )

        cell_record = {
            "coord": f"{col_letter}{r}",
            "col_idx": c,
            "col_letter": col_letter,
            "value": val,
        }
        if raw_frm:
          cell_record["formula"] = raw_frm
        if comment:
          cell_record["comment"] = comment
        if merge_info:
          cell_record["merge"] = merge_info
        if is_bold:
          cell_record["bold"] = True
        if indent:
          cell_record["indent"] = indent
        if num_fmt:
          cell_record["number_format"] = num_fmt
        if is_hidden_col:
          cell_record["hidden_col"] = True

        row_cells.append(cell_record)

        # Compact Markdown representation
        disp = (
            str(val)
            if val is not None
            else (f"[MERGED:{merge_info['origin_value']}]" if merge_info else "")
        )
        if disp:
          if is_bold:
            disp = f"**{disp}**"
          if comment:
            disp += f" *(Note: {comment})*"
          md_row_tokens.append(f"`{col_letter}{r}`: {disp}")

      if has_row_content:
        row_obj = {"row_idx": r, "cells": row_cells}
        if is_hidden_row:
          row_obj["hidden_row"] = True
        rows_data.append(row_obj)

        prefix = f"Row {r}" + (" [HIDDEN]" if is_hidden_row else "")
        md_lines.append(f"- **{prefix}**: " + " | ".join(md_row_tokens))

    safe_name = f"{idx+1:02d}_{sanitize_filename(sheet_name)}"
    json_path = os.path.join(output_dir, f"{safe_name}.json")
    md_path = os.path.join(output_dir, f"{safe_name}.md")

    sheet_payload = {
        "sheet_index": idx,
        "sheet_name": sheet_name,
        "sheet_state": ws_val.sheet_state,
        "dimensions": {
            "max_row": ws_val.max_row,
            "max_col": ws_val.max_column,
            "grid_col_start": get_column_letter(min_col) if grid_cols else None,
            "grid_col_end": get_column_letter(max_col) if grid_cols else None,
            "non_empty_rows": len(rows_data),
        },
        "merged_ranges": merged_ranges,
        "rows": rows_data,
    }

    with open(json_path, "w", encoding="utf-8") as jf:
      json.dump(sheet_payload, jf, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as mf:
      mf.write("\n".join(md_lines))

    manifest["sheets"].append({
        "sheet_name": sheet_name,
        "json_file": f"{safe_name}.json",
        "md_file": f"{safe_name}.md",
        "non_empty_rows": len(rows_data),
    })

  with open(os.path.join(output_dir, "_manifest.json"), "w", encoding="utf-8") as mf:
    json.dump(manifest, mf, indent=2, ensure_ascii=False)

  return manifest


if __name__ == "__main__":
  source_file = DOC_DIR / "excel" / "plant-performance.xlsx"
  target_dir = XOUT_DIR / "excel" / "sheet"
  manifest = dump_workbook_for_llm(source_file, target_dir)
  print(f"Extracted {manifest['sheet_count']} sheets.")