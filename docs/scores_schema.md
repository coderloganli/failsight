# Failure-prediction scores schema

`failsight score` writes one table of failure-prediction scores, consumed by
[TestLens](https://github.com/coderloganli/testlens). It is written twice with
identical content:

- DuckDB table `failure_scores` in the FailSight database (`paths.database`).
- Parquet file at `paths.scores_path` (default `data/scores/failure_scores.parquet`).

The column list and types are defined in `failsight.scoring.SCORES_SCHEMA`; this
document and that constant change together.

## Grain

One row per **drive × day × model family**. A drive-day scored by four model
families appears four times, distinguished by `model_family`.

The key is (`serial_number`, `score_date`, `model_family`).

## Columns

| Column | DuckDB type | Description |
|---|---|---|
| `serial_number` | VARCHAR | Drive serial number, as in the Backblaze data. |
| `model` | VARCHAR | Drive model, as in the Backblaze data. |
| `score_date` | DATE | Day of the SMART snapshot the score is computed from. The score uses only data up to and including this day. |
| `model_family` | VARCHAR | One of `logistic_regression`, `random_forest`, `gradient_boosting`, `lstm`, `isolation_forest`, `autoencoder`. |
| `score` | DOUBLE | Failure score in [0, 1]; higher means more likely to fail. See `score_type`. |
| `score_type` | VARCHAR | `classifier_score` or `anomaly_percentile` (below). |
| `horizon_days` | INTEGER | Prediction horizon: the supervised label is "fails within the next `horizon_days` days, counting `score_date`". |
| `run_id` | VARCHAR | MLflow run id of the training run that produced the model, for looking up its parameters and metrics. |
| `scored_at` | TIMESTAMP | UTC time the score was computed. |

## Score types

- **`classifier_score`** (`logistic_regression`, `random_forest`,
  `gradient_boosting`, `lstm`): output of a classifier trained with class
  weighting and negative subsampling. It ranks drives by risk but is **not a
  calibrated probability**; compare scores within one `model_family` only.
- **`anomaly_percentile`** (`isolation_forest`, `autoencoder`): produced only for
  drive models with too few recorded failures for supervised training. The value
  is the fraction of that drive model's healthy training rows that were less
  anomalous, so 0.99 means more anomalous than 99% of them. It is not tied to
  `horizon_days`.

## Scopes

- `failsight score --scope test` scores the held-out test split (evaluation).
- `failsight score --scope latest` scores every drive on the last day of data
  (operational use).

Each run replaces the whole table.

## Example query

```sql
-- Drives most at risk on the latest day, by gradient boosting
SELECT serial_number, model, score
FROM failure_scores
WHERE model_family = 'gradient_boosting'
  AND score_date = (SELECT max(score_date) FROM failure_scores)
ORDER BY score DESC
LIMIT 20;
```
