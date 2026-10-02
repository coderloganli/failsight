"""Unsupervised path for drive models with too few recorded failures.

For each such drive model, a scaler and PCA are fitted on its healthy training
rows (rows not within the label horizon of a failure); an Isolation Forest and a
small autoencoder are then fitted on the PCA components. Raw anomaly scores are
mapped to a percentile of the healthy training score distribution, so 0.99 means
"more anomalous than 99% of healthy training rows of this model".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch import nn

from failsight.data.split import LABEL_COLUMN
from failsight.models.common import set_seed, signed_log1p

ANOMALY_FAMILIES = ("isolation_forest", "autoencoder")


def low_failure_models(train: pd.DataFrame, min_failures: int) -> list[str]:
    """Drive models with fewer than ``min_failures`` drives labelled as failing in training."""
    failed = train[train[LABEL_COLUMN] == 1].groupby("model")["serial_number"].nunique()
    counts = failed.reindex(train["model"].unique(), fill_value=0)
    return sorted(counts[counts < min_failures].index)


class Autoencoder(nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_inputs, hidden_size), nn.Tanh())
        self.decoder = nn.Linear(hidden_size, n_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _percentile(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.searchsorted(reference, scores, side="right") / max(len(reference), 1)


@dataclass
class AnomalyModel:
    drive_model: str
    columns: list[str]
    reducer: Pipeline
    forest: IsolationForest
    autoencoder: Autoencoder
    reference: dict[str, np.ndarray] = field(default_factory=dict)

    def raw_scores(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        z = self.reducer.transform(frame[self.columns])
        with torch.no_grad():
            zt = torch.from_numpy(z.astype(np.float32))
            recon = ((self.autoencoder(zt) - zt) ** 2).mean(dim=1).numpy()
        return {"isolation_forest": -self.forest.score_samples(z), "autoencoder": recon}

    def score(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        raw = self.raw_scores(frame)
        return {name: _percentile(self.reference[name], raw[name]) for name in raw}


def fit_anomaly_model(
    train: pd.DataFrame, drive_model: str, columns: list[str], cfg: dict, seed: int
) -> AnomalyModel:
    set_seed(seed)
    acfg = cfg["anomaly"]
    healthy = train[(train["model"] == drive_model) & (train[LABEL_COLUMN] == 0)]
    columns = [c for c in columns if healthy[c].notna().any()]
    n_components = max(1, min(int(acfg["pca_components"]), len(columns), len(healthy) - 1))
    reducer = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(signed_log1p)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
        ]
    )
    z = reducer.fit_transform(healthy[columns])

    forest = IsolationForest(
        n_estimators=int(acfg["isolation_forest"]["n_estimators"]), random_state=seed
    ).fit(z)

    ae_cfg = acfg["autoencoder"]
    ae = Autoencoder(n_components, int(ae_cfg["hidden_size"]))
    optimizer = torch.optim.Adam(ae.parameters(), lr=float(ae_cfg["learning_rate"]))
    zt = torch.from_numpy(z.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(ae_cfg["batch_size"])
    ae.train()
    for _ in range(int(ae_cfg["epochs"])):
        order = torch.randperm(len(zt), generator=generator)
        for i in range(0, len(zt), batch_size):
            batch = zt[order[i : i + batch_size]]
            optimizer.zero_grad()
            loss = ((ae(batch) - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
    ae.eval()

    model = AnomalyModel(drive_model, columns, reducer, forest, ae)
    model.reference = {k: np.sort(v) for k, v in model.raw_scores(healthy).items()}
    return model
