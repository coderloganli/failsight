"""Helpers shared by the model families."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def usable_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Columns with at least one non-null value (drops attributes a fleet never reports)."""
    return [c for c in columns if df[c].notna().any()]


def drive_bucket(serials: pd.Series, n_buckets: int, seed: int = 0) -> np.ndarray:
    """Deterministic bucket in [0, n_buckets) per serial number."""

    def bucket(serial: str) -> int:
        digest = hashlib.md5(f"{serial}#{seed}".encode()).hexdigest()
        return int(digest[:8], 16) % n_buckets

    mapping = {s: bucket(s) for s in pd.unique(serials)}
    return serials.map(mapping).to_numpy()


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed SMART counters while keeping sign."""
    return np.sign(x) * np.log1p(np.abs(x))
