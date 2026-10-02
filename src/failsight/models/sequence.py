"""LSTM on raw (cleaned) SMART attribute sequences.

Each target drive-day is represented by the drive's last ``seq_len`` daily
observations up to and including that day (row-based, so gaps in reporting are
not padded), zero-padded at the front for drives with shorter history. Values
are signed-log transformed and standardized with training statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn

from failsight.config import smart_columns
from failsight.models.common import set_seed, signed_log1p


def load_sequence_frame(
    con: duckdb.DuckDBPyConnection, targets_sql: str, columns: list[str]
) -> pd.DataFrame:
    """Clean history for the drives in ``targets_sql`` (serial_number, date, label),
    up to each drive's last target date, with an ``is_target`` flag per row."""
    cols = ", ".join(f"c.{col}" for col in columns)
    return con.execute(
        f"""
        WITH t AS ({targets_sql}),
        last AS (SELECT serial_number, max(date) AS last_date FROM t GROUP BY 1)
        SELECT c.serial_number, c.model, c.date, {cols},
               t.serial_number IS NOT NULL AS is_target, t.label
        FROM clean c
        JOIN last USING (serial_number)
        LEFT JOIN t ON t.serial_number = c.serial_number AND t.date = c.date
        WHERE c.date <= last.last_date
        ORDER BY c.serial_number, c.date
        """
    ).df()


def build_sequences(
    values: np.ndarray, serials: np.ndarray, target_mask: np.ndarray, seq_len: int
) -> np.ndarray:
    """Return an array (n_targets, seq_len, n_features) of trailing windows.

    ``values`` rows must be sorted by serial then date.
    """
    serials = pd.Series(serials)
    pos = serials.groupby(serials, sort=False).cumcount().to_numpy()
    targets = np.flatnonzero(target_mask)
    out = np.zeros((len(targets), seq_len, values.shape[1]), dtype=np.float32)
    for k in range(seq_len):
        offset = seq_len - 1 - k
        valid = pos[targets] >= offset
        out[valid, k] = values[targets[valid] - offset]
    return out


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)


@dataclass
class SequenceModel:
    model: LSTMClassifier
    columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    seq_len: int

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = signed_log1p(frame[self.columns].to_numpy(dtype=float))
        values = (values - self.mean) / self.std
        return np.nan_to_num(values, nan=0.0).astype(np.float32)

    def sequences(self, frame: pd.DataFrame) -> np.ndarray:
        return build_sequences(
            self.transform(frame),
            frame["serial_number"].to_numpy(),
            frame["is_target"].to_numpy(dtype=bool),
            self.seq_len,
        )

    @torch.no_grad()
    def predict(self, frame: pd.DataFrame, batch_size: int = 2048) -> np.ndarray:
        self.model.eval()
        seqs = torch.from_numpy(self.sequences(frame))
        out = [
            torch.sigmoid(self.model(seqs[i : i + batch_size])).numpy()
            for i in range(0, len(seqs), batch_size)
        ]
        return np.concatenate(out) if out else np.empty(0)

    def save(self, path) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "columns": self.columns,
                "mean": self.mean,
                "std": self.std,
                "seq_len": self.seq_len,
                "hidden_size": self.model.lstm.hidden_size,
                "num_layers": self.model.lstm.num_layers,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> SequenceModel:
        state = torch.load(path, weights_only=False)
        model = LSTMClassifier(len(state["columns"]), state["hidden_size"], state["num_layers"])
        model.load_state_dict(state["state_dict"])
        return cls(model, state["columns"], state["mean"], state["std"], state["seq_len"])


def sequence_columns(cfg: dict, frame: pd.DataFrame | None = None) -> list[str]:
    cols = smart_columns(cfg)
    if frame is not None:
        cols = [c for c in cols if frame[c].notna().any()]
    return cols


def train_lstm(frame: pd.DataFrame, columns: list[str], cfg: dict, seed: int) -> SequenceModel:
    """Fit on the target rows of ``frame`` (labels in ``label``)."""
    set_seed(seed)
    lstm_cfg = cfg["lstm"]
    raw = signed_log1p(frame[columns].to_numpy(dtype=float))
    mean = np.nanmean(raw, axis=0)
    std = np.nanstd(raw, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.where(np.nan_to_num(std, nan=0.0) > 0, std, 1.0)
    model = LSTMClassifier(len(columns), lstm_cfg["hidden_size"], lstm_cfg["num_layers"])
    wrapper = SequenceModel(model, columns, mean, std, int(lstm_cfg["seq_len"]))

    X = torch.from_numpy(wrapper.sequences(frame))
    y = torch.from_numpy(frame.loc[frame["is_target"], "label"].to_numpy(dtype=np.float32))
    n_pos = float(y.sum())
    pos_weight = torch.tensor((len(y) - n_pos) / max(n_pos, 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lstm_cfg["learning_rate"]))
    batch_size = int(lstm_cfg["batch_size"])
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(int(lstm_cfg["epochs"])):
        order = torch.randperm(len(X), generator=generator)
        for i in range(0, len(X), batch_size):
            idx = order[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
    return wrapper
