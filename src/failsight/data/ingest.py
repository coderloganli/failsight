"""Load the raw daily CSVs into a typed ``snapshots`` table.

The set of SMART columns in the Backblaze CSVs changes over time, so files are
read with ``union_by_name`` and every value as text, then cast explicitly.
Configured SMART columns that are absent from every file become NULL.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from failsight.config import smart_columns


def _sql_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def ingest_snapshots(con: duckdb.DuckDBPyConnection, raw_dir: str | Path, cfg: dict) -> int:
    glob = _sql_path(Path(raw_dir) / "*.csv")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW raw_csv AS
        SELECT * FROM read_csv('{glob}', union_by_name = true, all_varchar = true, header = true)
        """
    )
    available = {row[0] for row in con.execute("DESCRIBE raw_csv").fetchall()}
    smart_exprs = [
        f"TRY_CAST({col} AS DOUBLE) AS {col}"
        if col in available
        else f"CAST(NULL AS DOUBLE) AS {col}"
        for col in smart_columns(cfg)
    ]
    con.execute(
        f"""
        CREATE OR REPLACE TABLE snapshots AS
        SELECT
            CAST(date AS DATE) AS date,
            serial_number,
            model,
            TRY_CAST(capacity_bytes AS BIGINT) AS capacity_bytes,
            CAST(failure AS INTEGER) AS failure,
            {", ".join(smart_exprs)}
        FROM raw_csv
        WHERE serial_number IS NOT NULL AND date IS NOT NULL
        """
    )
    return con.execute("SELECT count(*) FROM snapshots").fetchone()[0]
