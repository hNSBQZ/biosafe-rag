from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TABLE_RE = re.compile(r"<table\b.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
_TR_RE = re.compile(r"<tr\b.*?>.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b([^>]*)>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r"\b(rowspan|colspan)\s*=\s*['\"]?(\d+)", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _infer_table_name(markdown_text: str, table_start_index: int) -> str:
    """Infer a human-friendly table name from the text preceding a <table> block."""

    lookback = markdown_text[:table_start_index]
    for line in reversed(lookback.splitlines()[-50:]):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<"):
            continue
        return stripped
    return ""


@dataclass(frozen=True)
class HtmlCell:
    text: str
    rowspan: int = 1
    colspan: int = 1


def _clean_html_text(fragment: str) -> str:
    fragment = fragment.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    fragment = _TAG_RE.sub("", fragment)
    fragment = html.unescape(fragment)
    fragment = fragment.replace("\u00a0", " ")
    fragment = re.sub(r"[\t\r\f\v]+", " ", fragment)
    fragment = re.sub(r"\n+", "\n", fragment)
    fragment = re.sub(r" +", " ", fragment)
    return fragment.strip()


def _parse_cell_attrs(attr_text: str) -> tuple[int, int]:
    rowspan = 1
    colspan = 1
    for key, value in _ATTR_RE.findall(attr_text or ""):
        if key.lower() == "rowspan":
            rowspan = max(1, int(value))
        elif key.lower() == "colspan":
            colspan = max(1, int(value))
    return rowspan, colspan


def _parse_table_rows(table_html: str) -> list[list[HtmlCell]]:
    rows: list[list[HtmlCell]] = []
    for tr in _TR_RE.findall(table_html):
        cells: list[HtmlCell] = []
        for attr_text, inner_html in _CELL_RE.findall(tr):
            rowspan, colspan = _parse_cell_attrs(attr_text)
            cells.append(HtmlCell(text=_clean_html_text(inner_html), rowspan=rowspan, colspan=colspan))
        if cells:
            rows.append(cells)
    return rows


def _normalize_rowspans(rows: list[list[HtmlCell]]) -> list[list[str]]:
    """Expand rowspan/colspan into a rectangular-ish text grid.

    This is a best-effort normalizer for HTML tables. It does not try to infer
        semantic headers; it only expands spans so each row becomes a list of cell texts.

        Note:
        - HTML tables often use rowspan/colspan in multi-level headers.
        - This function duplicates the text into all covered cells so downstream logic
            can work with a simple 2D list of strings.
    """

    @dataclass
    class _Span:
        text: str
        remaining_rows: int

    grid: list[list[str]] = []
    spans: list[_Span] = []

    for row in rows:
        out_row: list[str] = []
        col = 0

        def ensure_spans_len(n: int) -> None:
            if n <= len(spans):
                return
            spans.extend([_Span(text="", remaining_rows=0) for _ in range(n - len(spans))])

        def flush_active_spans_until_free() -> None:
            nonlocal col
            ensure_spans_len(col + 1)
            while col < len(spans) and spans[col].remaining_rows > 0:
                out_row.append(spans[col].text)
                spans[col].remaining_rows -= 1
                col += 1
                ensure_spans_len(col + 1)

        for cell in row:
            flush_active_spans_until_free()

            cell_text = cell.text
            cell_rowspan = max(1, cell.rowspan)
            cell_colspan = max(1, cell.colspan)

            ensure_spans_len(col + cell_colspan)
            for offset in range(cell_colspan):
                out_row.append(cell_text)
                if cell_rowspan > 1:
                    spans[col + offset] = _Span(text=cell_text, remaining_rows=cell_rowspan - 1)
            col += cell_colspan

        # Fill remaining active spans on the right
        ensure_spans_len(col + 1)
        while col < len(spans) and spans[col].remaining_rows > 0:
            out_row.append(spans[col].text)
            spans[col].remaining_rows -= 1
            col += 1

        # Trim trailing empties (common with colspan-only headers)
        while out_row and out_row[-1] == "":
            out_row.pop()

        grid.append(out_row)

    return grid


def _is_left_header_table(grid: list[list[str]]) -> bool:
    """Detect tables where headers are in the leftmost column (transposed layout).

    Heuristic: few rows (2–5), significantly more columns than rows, and the
    first column contains non-numeric text labels (typically with CJK chars).
    """
    if not grid or len(grid) < 2 or len(grid) > 5:
        return False
    max_cols = max((len(r) for r in grid), default=0)
    if max_cols < len(grid) * 2:
        return False
    for row in grid:
        if not row or not row[0].strip():
            return False
        if re.fullmatch(r"\d+", row[0].strip()):
            return False
    return any(re.search(r"[\u4e00-\u9fff/\\%]", row[0]) for row in grid)


def _transpose_grid(grid: list[list[str]]) -> list[list[str]]:
    """Transpose a grid so rows become columns and vice versa."""
    if not grid:
        return grid
    max_cols = max((len(r) for r in grid), default=0)
    padded = [r + [""] * (max_cols - len(r)) for r in grid]
    return [[padded[r][c] for r in range(len(padded))] for c in range(max_cols)]


def _is_int_cell(value: str) -> bool:
    return bool(re.fullmatch(r"\d+", (value or "").strip()))


def _infer_header_row_count(grid: list[list[str]], max_scan_rows: int = 12) -> int:
    """Infer how many top rows are headers by finding the first data row.

    We want this to work without a hard-coded header keyword list.

    Heuristic (generic, but optimized for common "catalog" tables):
    - Scan from the top; the first row whose first cell is an integer is treated
      as the first data row.
    - Everything before that is considered header rows.
    - If we can't find such a row, fall back to treating the first row as header.

    This pairs well with `_normalize_rowspans`, because merged header text will
    already be expanded into repeated cells.
    """

    if not grid:
        return 0

    for i, row in enumerate(grid[:max_scan_rows]):
        if not row:
            continue
        first = row[0] if row else ""
        if _is_int_cell(first):
            return i

    # Fallback: assume the first row is a header if we cannot detect data rows.
    return 1


def _derive_columns(grid: list[list[str]]) -> tuple[list[str], int]:
    if not grid:
        return [], 0

    max_cols = max((len(r) for r in grid), default=0)
    if max_cols == 0:
        return [], 0

    header_count = _infer_header_row_count(grid)
    if header_count == 0:
        return [f"col_{i+1}" for i in range(max_cols)], 0

    columns: list[str] = []
    for c in range(max_cols):
        parts: list[str] = []
        for r in range(header_count):
            value = grid[r][c] if (r < len(grid) and c < len(grid[r])) else ""
            value = (value or "").strip()
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        col_name = "/".join(parts).strip() if parts else f"col_{c+1}"
        columns.append(col_name or f"col_{c+1}")

    # De-dupe column names if headers repeat
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for name in columns:
        count = seen.get(name, 0)
        if count == 0:
            deduped.append(name)
        else:
            deduped.append(f"{name}__{count+1}")
        seen[name] = count + 1

    return deduped, header_count


def _has_data_rowspan(parsed_rows: list[list[HtmlCell]], header_rows: int) -> bool:
    """Check whether any cell in the data rows has rowspan > 1."""
    for row in parsed_rows[header_rows:]:
        for cell in row:
            if cell.rowspan > 1:
                return True
    return False


def _aggregate_rowspan_groups(
    records: list[dict[str, str]], columns: list[str],
) -> list[dict[str, Any]]:
    """Merge consecutive records that share the same leading-column values (from rowspan).

    When only one column varies across the group, its values become a list.
    When multiple columns vary, they are collected as sub-dicts under a "子项" key.
    """
    if len(records) <= 1 or not columns:
        return records

    first_col = columns[0]

    groups: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = [records[0]]
    for rec in records[1:]:
        if rec.get(first_col) == current[0].get(first_col):
            current.append(rec)
        else:
            groups.append(current)
            current = [rec]
    groups.append(current)

    if all(len(g) == 1 for g in groups):
        return records

    result: list[dict[str, Any]] = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
            continue

        constant_cols: list[str] = []
        varying_cols: list[str] = []
        for col in columns:
            values = {rec.get(col, "") for rec in group}
            (constant_cols if len(values) <= 1 else varying_cols).append(col)

        if not varying_cols:
            result.append(group[0])
            continue

        aggregated: dict[str, Any] = {col: group[0].get(col, "") for col in constant_cols}
        if len(varying_cols) == 1:
            aggregated[varying_cols[0]] = [rec.get(varying_cols[0], "") for rec in group]
        else:
            aggregated["子项"] = [
                {col: rec.get(col, "") for col in varying_cols} for rec in group
            ]
        result.append(aggregated)

    return result


def parse_tables_from_text(text: str) -> list[dict[str, Any]]:
    """Parse all HTML tables in markdown text, return structured table data.

    Each returned dict contains:
        table_index, table_name, columns, header_rows, is_transposed,
        row_count, record_count, records, start_char
    """
    table_matches = list(_TABLE_RE.finditer(text))
    tables_out: list[dict[str, Any]] = []

    for table_index, match in enumerate(table_matches, start=1):
        table_html = match.group(0)
        table_name = _infer_table_name(text, match.start())
        parsed_rows = _parse_table_rows(table_html)
        normalized_grid = _normalize_rowspans(parsed_rows)

        is_transposed = _is_left_header_table(normalized_grid)
        if is_transposed:
            normalized_grid = _transpose_grid(normalized_grid)

        columns, header_rows = _derive_columns(normalized_grid)

        records: list[dict[str, str]] = []
        for row in normalized_grid[header_rows:]:
            if not any((v or "").strip() for v in row):
                continue
            row_values = list(row) + [""] * max(0, len(columns) - len(row))
            row_values = row_values[: len(columns)]

            first_value = (row_values[0] if row_values else "").strip()
            non_empty_indexes = [
                i for i, v in enumerate(row_values) if (v or "").strip()
            ]
            if records and first_value == "" and 0 not in non_empty_indexes and len(non_empty_indexes) <= 2:
                prev = records[-1]
                for i in non_empty_indexes:
                    key = columns[i]
                    addition = (row_values[i] or "").strip()
                    if not addition:
                        continue
                    existing = (prev.get(key) or "").strip()
                    prev[key] = f"{existing}\n{addition}" if existing else addition
                continue

            record = {columns[i]: (row_values[i] or "").strip() for i in range(len(columns))}
            records.append(record)

        if not is_transposed and _has_data_rowspan(parsed_rows, header_rows):
            records = _aggregate_rowspan_groups(records, columns)

        tables_out.append(
            {
                "table_index": table_index,
                "table_name": table_name,
                "columns": columns,
                "header_rows": header_rows,
                "is_transposed": is_transposed,
                "row_count": len(normalized_grid),
                "record_count": len(records),
                "records": records,
                "start_char": match.start(),
            }
        )

    return tables_out


def extract_tables_to_json(md_path: Path, out_dir: Path) -> dict[str, Any]:
    text = md_path.read_text(encoding="utf-8")
    tables_out = parse_tables_from_text(text)

    # Build raw per-tr rows for diagnostics
    table_matches = list(_TABLE_RE.finditer(text))
    raw_rows: list[dict[str, Any]] = []
    for table_index, match in enumerate(table_matches, start=1):
        parsed_rows = _parse_table_rows(match.group(0))
        for row_index, row_cells in enumerate(parsed_rows, start=1):
            raw_rows.append(
                {
                    "table_index": table_index,
                    "row_index": row_index,
                    "cells": [c.text for c in row_cells],
                }
            )

    # Strip internal field before writing
    clean_tables = [{k: v for k, v in t.items() if k != "start_char"} for t in tables_out]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mulu_raw_tr_rows.json").write_text(
        json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "mulu_tables_records.json").write_text(
        json.dumps(clean_tables, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_tr = sum(len(_parse_table_rows(m.group(0))) for m in table_matches)
    total_records = sum(t["record_count"] for t in tables_out)

    return {
        "tables": len(table_matches),
        "total_tr_rows": total_tr,
        "total_records": total_records,
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HTML tables in 名录.md into JSON.")
    parser.add_argument(
        "--input",
        default="名录.md",
        help="Input markdown file (default: 名录.md in current directory)",
    )
    parser.add_argument(
        "--out",
        default="out",
        help="Output directory (default: ./out)",
    )
    args = parser.parse_args()

    md_path = Path(args.input)
    out_dir = Path(args.out)

    if not md_path.exists():
        raise SystemExit(f"Input file not found: {md_path}")

    summary = extract_tables_to_json(md_path=md_path, out_dir=out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()