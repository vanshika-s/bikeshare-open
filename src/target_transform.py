"""Compare target transformations for gradient boosting."""

import json

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import ROOT, PROCESSED, TARGET
from src.features import feature_columns
from src.train import temporal_split


def evaluate(actual, predicted):
    return {
        "r2": round(float(r2_score(actual, predicted)), 4),
        "mae": round(float(mean_absolute_error(actual, predicted)), 1),
        "mean_predicted": round(float(np.mean(predicted)), 1),
        "mean_actual": round(float(np.mean(actual)), 1),
        "mean_error": round(float(np.mean(predicted - actual)), 1),
    }


def run_experiment():
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, _ = temporal_split(df)

    X_train = train[cols]
    X_valid = valid[cols]

    y_train = train[TARGET].to_numpy(dtype=float)
    y_valid = valid[TARGET].to_numpy(dtype=float)

    lag_train = train["cnt_lag_1"].to_numpy(dtype=float)
    lag_valid = valid["cnt_lag_1"].to_numpy(dtype=float)

    if np.any(y_train <= 0) or np.any(lag_train <= 0):
        raise ValueError("Training counts and lags must be positive")

    if np.any(lag_valid <= 0):
        raise ValueError("Validation lags must be positive")

    results = {
        "n_features": len(cols),
        "n_train": len(train),
        "n_valid": len(valid),
        "target_stats": {
            "train_min": float(y_train.min()),
            "train_max": float(y_train.max()),
            "valid_min": float(y_valid.min()),
            "valid_max": float(y_valid.max()),
            "log_train_min": round(float(np.log1p(y_train).min()), 4),
            "log_train_max": round(float(np.log1p(y_train).max()), 4),
            "log_valid_min": round(float(np.log1p(y_valid).min()), 4),
            "log_valid_max": round(float(np.log1p(y_valid).max()), 4),
        },
        "n_valid_exceeding_max_train": int(
            (y_valid > y_train.max()).sum()
        ),
        "approaches": {},
    }

    model = HistGradientBoostingRegressor(
        random_state=0, max_iter=300
    )
    model.fit(X_train, y_train)
    pred_original = model.predict(X_valid)

    model = HistGradientBoostingRegressor(
        random_state=0, max_iter=300
    )
    model.fit(X_train, np.log1p(y_train))
    pred_log = np.expm1(model.predict(X_valid))

    model = HistGradientBoostingRegressor(
        random_state=0, max_iter=300
    )
    model.fit(X_train, np.log(y_train / lag_train))
    pred_growth = np.exp(model.predict(X_valid)) * lag_valid

    results["approaches"] = {
        "original": evaluate(y_valid, pred_original),
        "log": evaluate(y_valid, pred_log),
        "relative_growth": evaluate(y_valid, pred_growth),
    }

    output = ROOT / "reports" / "target_transform.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
