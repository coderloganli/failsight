import duckdb
import numpy as np
import pandas as pd
import pytest

from failsight.config import load_config
from failsight.data.features import build_features


@pytest.fixture
def features():
    cfg = load_config(
        overrides={
            "data": {
                "raw_attributes": [5],
                "normalized_attributes": [],
                "cumulative_attributes": [5],
                "label_horizon_days": 2,
            },
            "features": {"windows_days": [3]},
        }
    )
    clean = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-06"]
            ).date,
            "serial_number": "A",
            "model": "M",
            "capacity_bytes": 4_000_000_000_000,
            "failure": [0, 0, 0, 0, 1],
            "failure_date": pd.Timestamp("2026-01-06").date(),
            "smart_5_raw": [0.0, 0.0, 1.0, 3.0, 6.0],
        }
    )
    con = duckdb.connect()
    con.register("clean_df", clean)
    con.execute("CREATE TABLE clean AS SELECT * FROM clean_df")
    build_features(con, cfg)
    return con.execute("SELECT * FROM features ORDER BY date").df()


def test_window_delta_uses_calendar_days(features):
    # 3-day windows: [01], [01-02], [01-03], [03-05] (04 missing), [05-06]
    assert features["smart_5_raw_delta_3d"].tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]


def test_rate_of_change_divides_by_elapsed_days(features):
    rate = features["smart_5_raw_rate_3d"].tolist()
    assert np.isnan(rate[0])  # no elapsed time
    assert rate[1:] == [0.0, 0.5, 1.0, 3.0]


def test_age_and_capacity(features):
    assert features["age_days"].tolist() == [0, 1, 2, 4, 5]
    assert features["capacity_tb"].iloc[0] == pytest.approx(4.0)


def test_label_marks_rows_within_horizon(features):
    # horizon 2 days counting today: failure day and the day before
    assert features["fails_within_horizon"].tolist() == [0, 0, 0, 1, 1]
