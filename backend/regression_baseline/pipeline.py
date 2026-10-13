"""
Baseline regression pipeline.

Loads the monthly levels panel from data.py, builds 8 features and 6 targets,
fits one OLS regression per target on a chronological 80/20 split, and writes
metrics, coefficients, predictions, and a feature-correlation diagnostic to
results/.

All modeling decisions live in this file. To change feature/target definitions
or the train/test split, edit here only. LinearRegression is deterministic
(closed-form OLS), so no random seed is needed.

Run from backend/ as:
    python -m regression_baseline.pipeline
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; do not open a GUI window
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression_baseline.data import load_monthly_panel


RESULTS_DIR = Path(__file__).parent / "results"
TEST_FRACTION = 0.2

# Each entry maps a feature name to (source column in the levels panel,
# transform). Transforms:
#   "diff"    = first difference; for things measured in percentage points
#               (rates, yields, unemployment rate).
#   "pct"     = month-over-month percentage change; for prices and indices.
#   "pct_yoy" = 12-month percentage change; the canonical YoY inflation read.
#   "level"   = identity; for series already constructed as deviations
#               from trend (e.g. CFNAI is already mean-zero, so differencing
#               it would be over-differencing).
FEATURE_DEFS = {
    "fed_funds_change": ("fed_funds", "diff"),
    "cpi_surprise":     ("cpi",       "pct_yoy"),
    "oil_change":       ("oil",       "pct"),
    "gdp_surprise":     ("gdp",       "pct"),
    "unrate_change":    ("unrate",    "diff"),
    "pmi":              ("cfnai",     "level"),
    "dxy_change":       ("dxy",       "pct"),
    "vix_change":       ("vix",       "pct"),
}

TARGET_DEFS = {
    "sp500_return":  ("sp500",  "pct"),
    "nasdaq_return": ("nasdaq", "pct"),
    "dgs10_change":  ("dgs10",  "diff"),
    "gold_return":   ("gold",   "pct"),
    "oil_return":    ("oil",    "pct"),
    "dxy_return":    ("dxy",    "pct"),
}

# Features to exclude when fitting a particular target. Two reasons appear:
#   (1) Algebraic identity: feature and target are computed from the same
#       underlying series (oil, dxy).
#   (2) Near-tautology: feature is mechanically constructed from the target
#       (VIX is implied volatility from SP500 options; NASDAQ tracks SP500
#       closely enough that the same concern applies).
EXCLUDED_FEATURES = {
    "sp500_return":  ["vix_change"],
    "nasdaq_return": ["vix_change"],
    "oil_return":    ["oil_change"],
    "dxy_return":    ["dxy_change"],
}


def _transform(series: pd.Series, kind: str) -> pd.Series:
    if kind == "pct":
        return series.pct_change()
    if kind == "pct_yoy":
        return series.pct_change(periods=12)
    if kind == "diff":
        return series.diff()
    if kind == "level":
        return series
    raise ValueError(f"Unknown transform: {kind}")


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    cols = {name: _transform(panel[src], kind) for name, (src, kind) in FEATURE_DEFS.items()}
    return pd.DataFrame(cols, index=panel.index)


def build_targets(panel: pd.DataFrame) -> pd.DataFrame:
    cols = {name: _transform(panel[src], kind) for name, (src, kind) in TARGET_DEFS.items()}
    return pd.DataFrame(cols, index=panel.index)


def train_and_evaluate(X: pd.DataFrame, y: pd.DataFrame):
    n = len(X)
    split = int(n * (1 - TEST_FRACTION))

    metrics = {}
    coefficients = {}
    predictions_long = []

    for target in y.columns:
        excluded = EXCLUDED_FEATURES.get(target, [])
        feat_cols = [c for c in X.columns if c not in excluded]
        X_t = X[feat_cols]
        y_t = y[target]

        X_train, X_test = X_t.iloc[:split], X_t.iloc[split:]
        y_train, y_test = y_t.iloc[:split], y_t.iloc[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics[target] = {
            "train_r2":   float(r2_score(y_train, y_train_pred)),
            "test_r2":    float(r2_score(y_test, y_test_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "test_rmse":  float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "train_mae":  float(mean_absolute_error(y_train, y_train_pred)),
            "test_mae":   float(mean_absolute_error(y_test, y_test_pred)),
            "n_train":    int(len(X_train)),
            "n_test":     int(len(X_test)),
            "n_features": int(len(feat_cols)),
            "excluded_features": excluded,
        }

        coef_row = {"intercept": float(model.intercept_)}
        for feat, coef in zip(feat_cols, model.coef_):
            coef_row[feat] = float(coef)
        for feat in excluded:
            coef_row[feat] = None
        coefficients[target] = coef_row

        for idx, actual, pred in zip(X_train.index, y_train.values, y_train_pred):
            predictions_long.append({
                "date": idx.date().isoformat(),
                "target": target,
                "actual": float(actual),
                "predicted": float(pred),
                "set": "train",
            })
        for idx, actual, pred in zip(X_test.index, y_test.values, y_test_pred):
            predictions_long.append({
                "date": idx.date().isoformat(),
                "target": target,
                "actual": float(actual),
                "predicted": float(pred),
                "set": "test",
            })

    return metrics, coefficients, predictions_long


def save_correlation_diagnostic(X: pd.DataFrame):
    corr = X.corr()
    corr.to_csv(RESULTS_DIR / "feature_correlations.csv", float_format="%.3f")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Feature correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_correlations.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {RESULTS_DIR / 'feature_correlations.csv'}")
    print(f"Wrote {RESULTS_DIR / 'feature_correlations.png'}")


def save_results(metrics, coefficients, predictions_long):
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {RESULTS_DIR / 'metrics.json'}")

    coef_df = pd.DataFrame(coefficients).T
    coef_df.index.name = "target"
    coef_df.to_csv(RESULTS_DIR / "coefficients.csv")
    print(f"Wrote {RESULTS_DIR / 'coefficients.csv'}")

    pd.DataFrame(predictions_long).to_csv(RESULTS_DIR / "predictions.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    panel = load_monthly_panel()
    print(f"Loaded panel: {panel.shape}")

    X_full = build_features(panel).dropna()
    y_full = build_targets(panel).dropna()
    common = X_full.index.intersection(y_full.index)
    X, y = X_full.loc[common], y_full.loc[common]
    print(f"Feature matrix X: {X.shape}, target matrix y: {y.shape}")
    print(f"Modeling window: {X.index.min().date()} -> {X.index.max().date()}")

    metrics, coefficients, predictions_long = train_and_evaluate(X, y)
    RESULTS_DIR.mkdir(exist_ok=True)
    save_results(metrics, coefficients, predictions_long)
    save_correlation_diagnostic(X)

    print("\n=== Summary (test set) ===")
    for target, m in metrics.items():
        excl = ",".join(m["excluded_features"]) or "-"
        print(f"  {target:15s}  test R²={m['test_r2']:+.3f}  test RMSE={m['test_rmse']:.4f}  train R²={m['train_r2']:+.3f}  (n={m['n_train']}/{m['n_test']}, k={m['n_features']}, excl={excl})")
