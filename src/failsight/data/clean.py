"""Clean the snapshots into a per-drive daily ``clean`` table.

Steps, all in SQL:

1. Deduplicate (serial_number, date), keeping a failure record if one exists.
2. Truncate each drive's history at its first failure; later rows for the same
   serial (re-used or re-installed drives) are dropped.
3. Treat negative SMART values as missing, then forward-fill missing values from
   the same drive's most recent observation.
4. Correct counter resets on cumulative attributes: when a counter drops below
   its previous value, the previous value is added as an offset to it and every
   later reading, so the corrected series keeps increasing.
"""

from __future__ import annotations

import duckdb

from failsight.config import smart_columns

_DRIVE_ORDER = "PARTITION BY serial_number ORDER BY date"
_UNTIL_NOW = f"({_DRIVE_ORDER} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
_META = "date, serial_number, model, capacity_bytes, failure, failure_date"


def clean_sql(cfg: dict) -> str:
    cols = smart_columns(cfg)
    cumulative_names = {f"smart_{n}_raw" for n in cfg["data"]["cumulative_attributes"]}
    cumulative = [c for c in cols if c in cumulative_names]

    valid = ", ".join(f"CASE WHEN {c} >= 0 THEN {c} END AS {c}" for c in cols)
    filled = ", ".join(f"last_value({c} IGNORE NULLS) OVER {_UNTIL_NOW} AS {c}" for c in cols)
    prev = "".join(f", lag({c}) OVER ({_DRIVE_ORDER}) AS {c}__prev" for c in cumulative)
    offsets = "".join(
        f", coalesce(sum(CASE WHEN {c} < {c}__prev THEN {c}__prev ELSE 0 END) "
        f"OVER {_UNTIL_NOW}, 0) AS {c}__offset"
        for c in cumulative
    )
    final = ", ".join(f"{c} + {c}__offset AS {c}" if c in cumulative else c for c in cols)

    return f"""
    CREATE OR REPLACE TABLE clean AS
    WITH dedup AS (
        SELECT * FROM snapshots
        QUALIFY row_number() OVER (PARTITION BY serial_number, date ORDER BY failure DESC) = 1
    ),
    with_failure AS (
        SELECT *, min(CASE WHEN failure = 1 THEN date END)
                      OVER (PARTITION BY serial_number) AS failure_date
        FROM dedup
    ),
    truncated AS (
        SELECT {_META}, {valid}
        FROM with_failure
        WHERE failure_date IS NULL OR date <= failure_date
    ),
    filled AS (
        SELECT {_META}, {filled} FROM truncated
    ),
    with_prev AS (
        SELECT *{prev} FROM filled
    ),
    with_offset AS (
        SELECT *{offsets} FROM with_prev
    )
    SELECT {_META}, {final}
    FROM with_offset
    ORDER BY serial_number, date
    """


def build_clean(con: duckdb.DuckDBPyConnection, cfg: dict) -> int:
    con.execute(clean_sql(cfg))
    return con.execute("SELECT count(*) FROM clean").fetchone()[0]
