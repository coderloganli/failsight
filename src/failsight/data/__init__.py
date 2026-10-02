"""Data layer: SQL on DuckDB over the raw Backblaze daily snapshots."""

from failsight.data.build import build_dataset, connect
from failsight.data.split import ID_COLUMNS, LABEL_COLUMN, feature_columns, load_split

__all__ = [
    "ID_COLUMNS",
    "LABEL_COLUMN",
    "build_dataset",
    "connect",
    "feature_columns",
    "load_split",
]
