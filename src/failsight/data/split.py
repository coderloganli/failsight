"""Leakage-free train/test split, by time and by drive.

- Drives are assigned to the train or test group by a deterministic hash of the
  serial number, so no drive contributes rows to both sets.
- Test rows lie on or after ``test_start``. Train rows are those whose whole
  label window ends before ``test_start`` (``date + horizon - 1 < test_start``),
  so no training label looks into the held-out period.
- Test rows whose label window runs past the last day of data are dropped,
  because their labels are not yet known.
- Negative training rows are subsampled at ``train_negative_fraction``;
  positives are always kept.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pandas as pd

ID_COLUMNS = ["serial_number", "model", "date", "failure", "failure_date", "split"]
LABEL_COLUMN = "fails_within_horizon"


def _unit_hash(expr: str) -> str:
    """SQL expression mapping a string to a deterministic number in [0, 1)."""
    return f"(CAST(('0x' || substr(md5({expr}), 1, 8)) AS UBIGINT) / 4294967296.0)"


def resolve_test_start(con: duckdb.DuckDBPyConnection, cfg: dict) -> dt.date:
    configured = cfg["split"].get("test_start")
    if configured:
        return pd.Timestamp(configured).date()
    q = float(cfg["split"]["test_start_quantile"])
    lo, hi = con.execute("SELECT min(date), max(date) FROM features").fetchone()
    return lo + dt.timedelta(days=int((hi - lo).days * q))


def build_split(con: duckdb.DuckDBPyConnection, cfg: dict) -> dict:
    split_cfg = cfg["split"]
    horizon = int(cfg["data"]["label_horizon_days"])
    test_start = resolve_test_start(con, cfg)
    test_frac = float(split_cfg["test_drive_fraction"])
    neg_frac = float(split_cfg["train_negative_fraction"])
    seed = int(split_cfg["seed"])

    drive_u = _unit_hash(f"serial_number || '#{seed}'")
    row_u = _unit_hash(f"serial_number || '#' || CAST(date AS VARCHAR) || '#{seed}'")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE dataset AS
        WITH tagged AS (
            SELECT *,
                   CASE WHEN {drive_u} < {test_frac} THEN 'test' ELSE 'train' END
                       AS drive_group,
                   max(date) OVER () AS last_date
            FROM features
        )
        SELECT * EXCLUDE (drive_group, last_date), drive_group AS split
        FROM tagged
        WHERE (drive_group = 'train'
               AND date + {horizon - 1} < DATE '{test_start}'
               AND ({LABEL_COLUMN} = 1 OR {row_u} < {neg_frac}))
           OR (drive_group = 'test'
               AND date >= DATE '{test_start}'
               AND date + {horizon - 1} <= last_date)
        ORDER BY serial_number, date
        """
    )
    con.execute(
        f"CREATE OR REPLACE TABLE split_info AS "
        f"SELECT DATE '{test_start}' AS test_start, {horizon} AS horizon_days"
    )
    counts = dict(con.execute("SELECT split, count(*) FROM dataset GROUP BY split").fetchall())
    return {"test_start": test_start, "rows": counts}


def feature_columns(con: duckdb.DuckDBPyConnection, table: str = "dataset") -> list[str]:
    cols = [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]
    return [c for c in cols if c not in ID_COLUMNS and c != LABEL_COLUMN]


def load_split(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM dataset WHERE split = ? ORDER BY date, serial_number", [split]
    ).df()
