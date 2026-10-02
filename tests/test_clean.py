import duckdb
import numpy as np
import pandas as pd

from failsight.config import load_config
from failsight.data.clean import build_clean

from .synthetic import LATE_COLUMN, LATE_COLUMN_DAY, RESET_SERIAL, START


def _tiny_cfg():
    return load_config(
        overrides={
            "data": {
                "raw_attributes": [5, 241],
                "normalized_attributes": [],
                "cumulative_attributes": [241],
            }
        }
    )


def _clean(rows):
    con = duckdb.connect()
    snapshots = pd.DataFrame(
        rows, columns=["date", "serial_number", "failure", "smart_5_raw", "smart_241_raw"]
    ).assign(model="M", capacity_bytes=1)
    snapshots["date"] = pd.to_datetime(snapshots["date"]).dt.date
    con.register("snapshots_df", snapshots)
    con.execute("CREATE TABLE snapshots AS SELECT * FROM snapshots_df")
    build_clean(con, _tiny_cfg())
    return con.execute("SELECT * FROM clean ORDER BY serial_number, date").df()


def test_counter_reset_is_corrected_to_a_monotone_series():
    out = _clean(
        [
            ("2026-01-01", "A", 0, 0, 100),
            ("2026-01-02", "A", 0, 0, 150),
            ("2026-01-03", "A", 0, 0, 10),  # reset
            ("2026-01-04", "A", 0, 0, 30),
        ]
    )
    assert out["smart_241_raw"].tolist() == [100, 150, 160, 180]


def test_missing_and_negative_values_are_forward_filled_per_drive():
    out = _clean(
        [
            ("2026-01-01", "A", 0, 4, 1),
            ("2026-01-02", "A", 0, None, 2),
            ("2026-01-03", "A", 0, -1, 3),
            ("2026-01-01", "B", 0, None, 1),
            ("2026-01-02", "B", 0, 7, 2),
        ]
    )
    a = out[out["serial_number"] == "A"]["smart_5_raw"].tolist()
    b = out[out["serial_number"] == "B"]["smart_5_raw"].tolist()
    assert a == [4, 4, 4]
    assert np.isnan(b[0]) and b[1] == 7  # never filled from another drive


def test_history_is_truncated_at_first_failure_and_deduplicated():
    out = _clean(
        [
            ("2026-01-01", "A", 0, 0, 1),
            ("2026-01-02", "A", 0, 0, 2),
            ("2026-01-02", "A", 1, 0, 2),  # duplicate day, failure record wins
            ("2026-01-03", "A", 0, 0, 3),  # after failure: dropped
        ]
    )
    assert out["date"].astype(str).tolist() == ["2026-01-01", "2026-01-02"]
    assert out["failure"].tolist() == [0, 1]
    assert (out["failure_date"].astype(str) == "2026-01-02").all()


def test_synthetic_ingest_handles_schema_drift_duplicates_and_resets(con):
    snap = con.execute(f"SELECT date, {LATE_COLUMN} FROM snapshots ORDER BY date").df()
    early = snap["date"] < pd.Timestamp(START) + pd.Timedelta(days=LATE_COLUMN_DAY)
    assert snap.loc[early, LATE_COLUMN].isna().all()
    assert snap.loc[~early, LATE_COLUMN].notna().all()

    dupes = con.execute(
        "SELECT count(*) FROM (SELECT serial_number, date FROM clean GROUP BY 1, 2 "
        "HAVING count(*) > 1)"
    ).fetchone()[0]
    assert dupes == 0

    raw = con.execute(
        "SELECT smart_241_raw FROM snapshots WHERE serial_number = ? ORDER BY date",
        [RESET_SERIAL],
    ).df()["smart_241_raw"]
    fixed = con.execute(
        "SELECT smart_241_raw FROM clean WHERE serial_number = ? ORDER BY date", [RESET_SERIAL]
    ).df()["smart_241_raw"]
    assert (raw.diff().dropna() < 0).any()
    assert (fixed.diff().dropna() >= 0).all()
