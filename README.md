# FailSight

Predictive maintenance on the public Backblaze hard-drive dataset: daily SMART telemetry for a large fleet of data-center drives, where the task is to flag a drive days before it fails.

## Pipeline

- **Data layer**: SQL on DuckDB over the raw daily snapshots. Cleans vendor-specific SMART attributes, handles missing and reset counters, and splits train and test by time and by drive so no drive's future leaks into its own training data.
- **Exploratory analysis**: pandas with Matplotlib and seaborn plots of attribute distributions, failure rates by model and pre-failure trajectories, used to choose the features: rolling-window deltas, rates of change and drive age.
- **Supervised models**: scikit-learn logistic regression, random forest and gradient boosting under severe class imbalance. Hyperparameters are tuned with Optuna under time-based cross-validation, features are selected by permutation importance, and models are evaluated on precision-recall rather than accuracy.
- **Sequence model**: an LSTM in PyTorch on the raw attribute sequences, compared against the feature-engineered models on the same held-out period.
- **Unsupervised path**: for drive models with too few recorded failures to train on, PCA for dimensionality reduction, then Isolation Forest and an autoencoder scoring anomalous drives.
- **Experiment tracking**: every run's parameters and metrics are tracked in MLflow.

The failure-prediction scores produced here feed [TestLens](https://github.com/coderloganli/testlens), a GenAI workbench for querying the test data in plain language.

## Getting started

Requires Python 3.10+. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough
pip install -e ".[dev]"
```

Run the pipeline stages (settings live in `configs/default.yaml`; pass `--config my.yaml` to override any of them):

```bash
failsight download --periods Q1_2026   # fetch and extract Backblaze quarterly archives into data/raw
failsight build                        # DuckDB: ingest, clean, features, leakage-free split
failsight train                        # all model families; runs are logged to MLflow (sqlite:///mlflow.db)
failsight score --scope latest         # write the failure-prediction scores table
```

Browse runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. The EDA notebook is `notebooks/01_eda.ipynb`, and the scores table that TestLens reads is described in [docs/scores_schema.md](docs/scores_schema.md).

Tests run on a small synthetic SMART-like fixture and need no download:

```bash
ruff check . && pytest
```

## Status

Under active development. The scope above is committed; code is being added to this repository as it is built.
