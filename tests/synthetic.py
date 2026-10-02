"""Tiny synthetic SMART-like dataset in the Backblaze daily-CSV layout.

Not real data and not statistically representative: it exists only to exercise
the pipeline mechanics (schema drift, missing values, counter resets, failures).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

START = dt.date(2026, 1, 1)
N_DAYS = 120
# (model name, drives, failing drives)
MODELS = [("SYN-MODEL-A", 50, 10), ("SYN-MODEL-B", 16, 2)]
RESET_SERIAL = "SYN-MODEL-A-0003"
RESET_DAY = 50
LATE_COLUMN = "smart_242_raw"  # absent from the files before LATE_COLUMN_DAY
LATE_COLUMN_DAY = 30


def generate_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model, n_drives, n_fail in MODELS:
        for i in range(n_drives):
            serial = f"{model}-{i:04d}"
            first_day = int(rng.integers(0, 20))
            fail_day = int(rng.integers(45, N_DAYS - 5)) if i < n_fail else None
            last_day = fail_day if fail_day is not None else N_DAYS - 1
            hours = float(rng.integers(1_000, 30_000))
            lbas = float(rng.integers(1e9, 5e9))
            realloc = 0.0
            uncorrectable = 0.0
            for day in range(first_day, last_day + 1):
                hours += 24
                lbas += float(rng.integers(1e6, 5e6))
                if day == RESET_DAY and serial == RESET_SERIAL:
                    lbas = 1_000.0
                degrading = fail_day is not None and fail_day - day < 20
                if degrading:
                    realloc += float(rng.integers(1, 8))
                    uncorrectable += float(rng.integers(0, 3))
                rows.append(
                    {
                        "date": START + dt.timedelta(days=day),
                        "serial_number": serial,
                        "model": model,
                        "capacity_bytes": 8_001_563_222_016,
                        "failure": int(day == fail_day),
                        "smart_1_normalized": float(rng.integers(75, 120)),
                        "smart_1_raw": float(rng.integers(0, 2**31)),
                        "smart_5_raw": realloc,
                        "smart_9_raw": hours,
                        "smart_187_raw": uncorrectable,
                        "smart_194_raw": float(rng.normal(32, 3)),
                        "smart_197_raw": float(rng.integers(0, 4)) if degrading else 0.0,
                        "smart_241_raw": lbas,
                        "smart_242_raw": lbas * 1.5,
                    }
                )
    df = pd.DataFrame(rows)
    # Sprinkle missing values over a few attributes.
    for col in ["smart_5_raw", "smart_194_raw", "smart_187_raw"]:
        mask = rng.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan
    return df


def write_daily_csvs(df: pd.DataFrame, raw_dir: Path) -> list[Path]:
    """Write one CSV per day; early files lack LATE_COLUMN and one row is duplicated."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for date, day_df in df.groupby("date"):
        day_index = (date - START).days
        out = day_df
        if day_index < LATE_COLUMN_DAY:
            out = out.drop(columns=[LATE_COLUMN])
        if day_index == 10:
            out = pd.concat([out, out.iloc[[0]]])
        path = raw_dir / f"{date.isoformat()}.csv"
        out.to_csv(path, index=False)
        paths.append(path)
    return paths
