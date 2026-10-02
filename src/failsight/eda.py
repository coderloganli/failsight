"""Exploratory analysis helpers (pandas, Matplotlib, seaborn).

Aggregations run in DuckDB and come back as pandas frames; each plot function
returns a Matplotlib figure so notebooks and scripts can show or save it.
"""

from __future__ import annotations

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sample_drives(con: duckdb.DuckDBPyConnection, n_drives: int = 2000, seed: int = 0):
    """All clean rows for a random sample of drives, failed drives always included."""
    return con.execute(
        """
        WITH drives AS (
            SELECT serial_number, max(failure) AS failed FROM clean GROUP BY 1
        ),
        chosen AS (
            SELECT serial_number FROM drives WHERE failed = 1
            UNION
            SELECT serial_number FROM (
                SELECT serial_number FROM drives WHERE failed = 0
                ORDER BY hash(serial_number || CAST(? AS VARCHAR)) LIMIT ?
            )
        )
        SELECT c.*, d.failed
        FROM clean c JOIN chosen USING (serial_number) JOIN drives d USING (serial_number)
        ORDER BY serial_number, date
        """,
        [seed, n_drives],
    ).df()


def failure_rates_by_model(con: duckdb.DuckDBPyConnection, min_drive_days: int = 0):
    """Annualized failure rate per drive model: failures / (drive days / 365), in percent."""
    return con.execute(
        """
        SELECT model,
               count(DISTINCT serial_number) AS drives,
               count(*) AS drive_days,
               sum(failure) AS failures,
               100.0 * sum(failure) / (count(*) / 365.0) AS annualized_failure_rate_pct
        FROM clean
        GROUP BY model
        HAVING count(*) >= ?
        ORDER BY annualized_failure_rate_pct DESC
        """,
        [min_drive_days],
    ).df()


def prefailure_trajectories(con: duckdb.DuckDBPyConnection, column: str, days_before: int = 60):
    """Median and interquartile range of a SMART column by days before failure."""
    return con.execute(
        f"""
        SELECT date_diff('day', date, failure_date) AS days_to_failure,
               quantile_cont({column}, 0.25) AS q25,
               median({column}) AS median,
               quantile_cont({column}, 0.75) AS q75,
               count(*) AS n
        FROM clean
        WHERE failure_date IS NOT NULL
          AND date_diff('day', date, failure_date) BETWEEN 0 AND ?
        GROUP BY 1
        ORDER BY 1
        """,
        [days_before],
    ).df()


def plot_attribute_distributions(df: pd.DataFrame, columns: list[str], ncols: int = 3):
    """Histograms of log1p(value), failed vs healthy drives."""
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, columns, strict=False):
        data = df[[col, "failed"]].dropna()
        data = data.assign(value=np.log1p(data[col].clip(lower=0)))
        sns.histplot(
            data,
            x="value",
            hue="failed",
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel("log1p(value)")
    for ax in list(axes.flat)[len(columns) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_failure_rates(rates: pd.DataFrame, top_n: int = 20):
    top = rates.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top) + 1))
    sns.barplot(top, x="annualized_failure_rate_pct", y="model", ax=ax, color="steelblue")
    ax.set_xlabel("Annualized failure rate (%)")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_prefailure_trajectory(traj: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(traj["days_to_failure"], traj["median"], label="median")
    ax.fill_between(traj["days_to_failure"], traj["q25"], traj["q75"], alpha=0.3, label="IQR")
    ax.invert_xaxis()
    ax.set_xlabel("Days before failure")
    ax.set_ylabel(column)
    ax.legend()
    fig.tight_layout()
    return fig
