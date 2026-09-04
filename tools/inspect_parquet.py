#!/usr/bin/env python3
"""Inventory parquet files without mutating model data.

Prints path, size, row/column counts, schema, and a small head sample for every
parquet under a root. Intended for GitHub Actions and audit work.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def inspect(path: Path) -> None:
    rel = path.as_posix()
    size = path.stat().st_size
    meta = pq.read_metadata(path)
    print("=" * 100)
    print(f"FILE: {rel}")
    print(f"BYTES: {size:,}")
    print(f"ROWS: {meta.num_rows:,}")
    print(f"ROW_GROUPS: {meta.num_row_groups:,}")
    print(f"COLUMNS: {meta.num_columns:,}")
    print("SCHEMA:")
    print(meta.schema)
    try:
        table = pq.read_table(path).slice(0, 5)
        print("HEAD(5):")
        print(table.to_pandas().to_string(index=False))
    except Exception as exc:
        print(f"HEAD_READ_ERROR: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.parquet"))
    print(f"parquet files found: {len(files)}")
    for path in files:
        inspect(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
