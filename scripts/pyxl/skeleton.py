import re
from pathlib import Path
from scripts.constants import XOUT_DIR


ROW_LINE_RE = re.compile(r"^- \*\*Row (\d+)(\s*\[HIDDEN\])?\*\*:\s*(.+)$")
CELL_RE = re.compile(r"`([A-Z]+)(\d+)`:\s*([^|]+)")


def _is_numeric_literal(raw_val: str) -> bool:
    cleaned = raw_val.strip().strip("*").strip()
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def skeleton_from_markdown(md_text: str, header_cutoff_row: int = 4) -> str:
    """Stage 2: Compress a sheet markdown dump into a structural skeleton for the LLM."""
    out_lines: list[str] = []

    for line in md_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# Sheet:") or stripped.startswith("**Merged Ranges:**"):
            out_lines.append(stripped)
            continue

        match = ROW_LINE_RE.match(stripped)
        if not match:
            continue

        row_num = int(match.group(1))
        hidden_flag = match.group(2) or ""
        cells_blob = match.group(3)

        if row_num <= header_cutoff_row:
            out_lines.append(stripped)
            continue

        kept_cells: list[str] = []
        numeric_cols: list[str] = []

        for col_letter, _, raw_val in CELL_RE.findall(cells_blob):
            val = raw_val.strip()
            if col_letter in {"A", "B", "C"}:
                kept_cells.append(f"`{col_letter}{row_num}`: {val}")
            elif val.startswith("[MERGED:"):
                kept_cells.append(f"`{col_letter}{row_num}`: {val}")
            elif not _is_numeric_literal(val):
                kept_cells.append(f"`{col_letter}{row_num}`: {val}")
            else:
                numeric_cols.append(col_letter)

        if numeric_cols:
            span = (
                numeric_cols[0]
                if len(numeric_cols) == 1
                else f"{numeric_cols[0]}..{numeric_cols[-1]} ({len(numeric_cols)} cols)"
            )
            kept_cells.append(f"[has_values: {span}]")

        out_lines.append(f"- **Row {row_num}{hidden_flag}**: " + " | ".join(kept_cells))

    return "\n".join(out_lines)


def run_stage2(
    source_dir: Path,
    target_dir: Path,
    header_cutoff_row: int = 4,
) -> list[Path]:
    """Run Stage 2 over a directory of sheet *.md files -> *.skeleton.md."""
    
    outputs: list[Path] = []
    for md_path in sorted(source_dir.glob("*.md")):
        if md_path.name.endswith(".skeleton.md"):
            continue
        skeleton = skeleton_from_markdown(
            md_path.read_text(encoding="utf-8"),
            header_cutoff_row=header_cutoff_row,
        )
        out_path = target_dir / f"{md_path.stem}.skeleton.md"
        out_path.write_text(skeleton, encoding="utf-8")
        outputs.append(out_path)

    return outputs


def main():
    source_dir = XOUT_DIR / "excel" / "sheet"
    target_dir = XOUT_DIR / "excel" / "skeleton"
    run_stage2(source_dir=source_dir, target_dir=target_dir)


if __name__ == "__main__":
    main()
 
