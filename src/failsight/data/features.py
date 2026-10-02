"""Feature engineering in SQL over the ``clean`` table.

For each SMART column and each trailing window of ``w`` days (the current day
and the ``w - 1`` days before it, by calendar date):

- ``<col>_delta_<w>d``: current value minus the first value in the window.
- ``<col>_rate_<w>d``: that delta divided by the days elapsed in the window.

Plus drive age: ``age_days`` (days since the drive first appears in the data,
left-censored at the start of the data) and ``power_on_days`` (SMART 9 / 24).

Every feature uses only the drive's own past, so features can be computed on the
full history before the train/test split. The label ``fails_within_horizon`` is
1 when the drive fails within the next ``label_horizon_days`` days, counting today.
"""

from __future__ import annotations

import duckdb

from failsight.config import smart_columns


def feature_sql(cfg: dict) -> str:
    cols = smart_columns(cfg)
    windows = [int(w) for w in cfg["features"]["windows_days"]]
    horizon = int(cfg["data"]["label_horizon_days"])

    exprs = list(cols)
    for w in windows:
        for c in cols:
            delta = f"({c} - first_value({c} IGNORE NULLS) OVER w{w})"
            elapsed = f"date_diff('day', min(date) OVER w{w}, date)"
            exprs.append(f"{delta} AS {c}_delta_{w}d")
            exprs.append(f"{delta} / NULLIF({elapsed}, 0) AS {c}_rate_{w}d")
    exprs.append("date_diff('day', min(date) OVER (PARTITION BY serial_number), date) AS age_days")
    if "smart_9_raw" in cols:
        exprs.append("smart_9_raw / 24.0 AS power_on_days")
    exprs.append("capacity_bytes / 1e12 AS capacity_tb")

    window_clause = ""
    if windows:
        window_clause = "WINDOW " + ", ".join(
            f"w{w} AS (PARTITION BY serial_number ORDER BY date "
            f"RANGE BETWEEN INTERVAL {w - 1} DAYS PRECEDING AND CURRENT ROW)"
            for w in windows
        )
    return f"""
    CREATE OR REPLACE TABLE features AS
    SELECT
        serial_number,
        model,
        date,
        failure,
        failure_date,
        CAST(failure_date IS NOT NULL
             AND date_diff('day', date, failure_date) < {horizon} AS INTEGER)
            AS fails_within_horizon,
        {", ".join(exprs)}
    FROM clean
    {window_clause}
    ORDER BY serial_number, date
    """


def build_features(con: duckdb.DuckDBPyConnection, cfg: dict) -> int:
    con.execute(feature_sql(cfg))
    return con.execute("SELECT count(*) FROM features").fetchone()[0]
