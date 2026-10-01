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

## Status

Under active development. The scope above is committed; code is being added to this repository as it is built.
