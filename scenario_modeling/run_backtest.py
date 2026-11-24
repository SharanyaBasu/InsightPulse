"""
run_backtest.py  —  Task 2: Historical backtesting for InsightPulse Ridge model.

Evaluates Ridge predictions across historical dates using only data available
at each date. Computes actual 21-trading-day forward returns and compares
against model baseline predictions (neutral scenario, all shocks = 0).

Usage (from repo root):
    python scenario_modeling/run_backtest.py \
        --bundle scenario_modeling/artifacts/scenario-playground-ridge-v1.0.0/v1.0.0 \
        --db backend/insightpulse.db \
        --out scenario_modeling/backtest_results.csv
"""

import argparse
import csv
import math
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from src.scenario_runtime import load_scenario_model_bundle

HORIZON = 21  # trading days

# Maps each model target to (db_column, return_type)
TARGET_COLUMN_MAP = {
    "sp500":                  ("sp500",      "pct"),
    "nasdaq":                 ("nasdaq",     "pct"),
    "ten_year_yield_bps":     ("tnx",        "bps"),
    "dxy":                    ("usd_index",  "pct"),
    "gold":                   ("gold",       "pct"),
    "oil":                    ("oil",        "pct"),
    "technology":             ("xlk",        "pct"),
    "energy":                 ("xle",        "pct"),
    "financials":             ("xlf",        "pct"),
    "utilities":              ("xlp",        "pct"),   # XLU not in DB; XLP proxy
    "healthcare":             ("xlv",        "pct"),
    "consumer_discretionary": ("xly",        "pct"),
}

NEUTRAL_SHOCKS = {
    "fed_funds_rate_change_bps": 0.0,
    "cpi_surprise_pct":          0.0,
    "oil_price_change_pct":      0.0,
    "gdp_growth_surprise_pct":   0.0,
    "unemployment_change_pct":   0.0,
    "pmi_change_points":         0.0,
    "dxy_change_pct":            0.0,
    "vix_change_points":         0.0,
}


def load_market_rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM market_data ORDER BY date")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    # Also try to load macro data
    macro = {}
    try:
        cur2 = conn.cursor()
        cur2.execute("SELECT * FROM macro_data ORDER BY date")
        for r in cur2.fetchall():
            macro[r["date"]] = dict(r)
    except Exception:
        pass

    print(f"Loaded {len(rows)} market_data rows from {db_path}")
    return rows, macro


def safe_float(value):
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def series_values(rows, col, up_to_idx):
    """Return list of valid floats for col from rows[0..up_to_idx] inclusive."""
    vals = []
    for r in rows[:up_to_idx + 1]:
        v = safe_float(r.get(col))
        if v is not None:
            vals.append(v)
    return vals


def pct_return(current, previous):
    if previous and previous != 0:
        return ((current / previous) - 1) * 100
    return None


def build_state_vector(rows, macro, idx, state_feature_names):
    """
    Build the 45-feature state vector for the model at rows[idx].
    Feature names come directly from the bundle.
    """
    row = rows[idx]
    state = {}

    def s(col):
        return series_values(rows, col, idx)

    # --- Macro features ---
    # These come from macro_data table if available, else we approximate from market data
    # We use the most recent macro row on or before this market date
    row_date = row["date"]
    macro_row = None
    for d in sorted(macro.keys(), reverse=True):
        if d <= row_date:
            macro_row = macro[d]
            break

    if macro_row:
        state["fed_funds_target_midpoint_pct"] = safe_float(macro_row.get("fed_funds_rate")) or 0.0
        state["effective_fed_funds_rate_pct"]  = safe_float(macro_row.get("fed_funds_rate")) or 0.0
        state["cpi_mom_pct"]                   = 0.0   # not directly available; default
        state["cpi_yoy_pct"]                   = safe_float(macro_row.get("cpi")) or 0.0
        state["unemployment_rate_pct"]          = safe_float(macro_row.get("unemployment")) or 0.0
        state["real_gdp_qoq_annualized_pct"]   = safe_float(macro_row.get("gdp")) or 0.0
        state["ten_year_yield_pct"]             = safe_float(macro_row.get("ten_year_yield")) or safe_float(row.get("tnx")) or 0.0
        state["two_year_yield_pct"]             = safe_float(macro_row.get("two_year_yield")) or safe_float(row.get("irx")) or 0.0
    else:
        # Fallback to market data approximations
        state["fed_funds_target_midpoint_pct"] = safe_float(row.get("irx")) or 0.0
        state["effective_fed_funds_rate_pct"]  = safe_float(row.get("irx")) or 0.0
        state["cpi_mom_pct"]                   = 0.0
        state["cpi_yoy_pct"]                   = 0.0
        state["unemployment_rate_pct"]         = 0.0
        state["real_gdp_qoq_annualized_pct"]   = 0.0
        state["ten_year_yield_pct"]            = safe_float(row.get("tnx")) or 0.0
        state["two_year_yield_pct"]            = safe_float(row.get("irx")) or 0.0

    # Yield curve slope and real rate
    state["yield_curve_slope_10y_2y_pct_points"] = state["ten_year_yield_pct"] - state["two_year_yield_pct"]
    state["real_rate_proxy_pct_points"]          = state["effective_fed_funds_rate_pct"] - state["cpi_yoy_pct"]

    # VIX
    state["vix_level_points"] = safe_float(row.get("vix")) or 20.0

    # --- Market derived features ---
    assets = {
        "sp500":  "sp500",
        "nasdaq": "nasdaq",
        "gold":   "gold",
        "oil":    "oil",
        "dxy":    "usd_index",
    }

    for name, col in assets.items():
        vals = s(col)
        if not vals:
            # Fill all features for this asset with 0
            for suffix in ["return_5d_pct", "return_21d_pct", "volatility_20d_annualized_pct",
                           "drawdown_252d_pct", "ma_distance_50d_pct", "ma_distance_200d_pct", "zscore_252d"]:
                key = f"{name}_{suffix}"
                if key in state_feature_names:
                    state[key] = 0.0
            continue

        v = vals[-1]

        # 5d return
        key5 = f"{name}_return_5d_pct"
        if key5 in state_feature_names:
            state[key5] = pct_return(v, vals[-6]) if len(vals) >= 6 else 0.0

        # 21d return
        key21 = f"{name}_return_21d_pct"
        if key21 in state_feature_names:
            state[key21] = pct_return(v, vals[-22]) if len(vals) >= 22 else 0.0

        # 20d annualised vol
        keyvol = f"{name}_volatility_20d_annualized_pct"
        if keyvol in state_feature_names:
            if len(vals) >= 21 and all(x > 0 for x in vals[-21:]):
                log_rets = [math.log(vals[i] / vals[i-1]) for i in range(len(vals)-20, len(vals))]
                state[keyvol] = statistics.stdev(log_rets) * math.sqrt(252) * 100
            else:
                state[keyvol] = 0.0

        # 252d drawdown
        keydd = f"{name}_drawdown_252d_pct"
        if keydd in state_feature_names:
            window252 = vals[-252:] if len(vals) >= 252 else vals
            peak = max(window252)
            state[keydd] = ((v - peak) / peak * 100) if peak != 0 else 0.0

        # MA distances
        for ma_w, suffix in [(50, "ma_distance_50d_pct"), (200, "ma_distance_200d_pct")]:
            key_ma = f"{name}_{suffix}"
            if key_ma in state_feature_names:
                if len(vals) >= ma_w:
                    ma = statistics.mean(vals[-ma_w:])
                    state[key_ma] = ((v - ma) / ma * 100) if ma != 0 else 0.0
                else:
                    state[key_ma] = 0.0

        # 252d z-score
        keyz = f"{name}_zscore_252d"
        if keyz in state_feature_names:
            z_vals = vals[-252:] if len(vals) >= 252 else (vals if len(vals) >= 60 else [])
            if z_vals:
                std = statistics.stdev(z_vals)
                state[keyz] = ((v - statistics.mean(z_vals)) / std) if std != 0 else 0.0
            else:
                state[keyz] = 0.0

    # Fill any remaining features with 0 as fallback
    for feat in state_feature_names:
        if feat not in state:
            state[feat] = 0.0

    return state


def compute_actual_return(rows, idx, target):
    col, kind = TARGET_COLUMN_MAP[target]
    current = safe_float(rows[idx].get(col))
    if current is None:
        return None
    future_idx = idx + HORIZON
    if future_idx >= len(rows):
        return None
    future = safe_float(rows[future_idx].get(col))
    if future is None:
        return None
    if kind == "pct":
        return pct_return(future, current)
    else:  # bps
        return (future - current) * 100


def run_backtest(bundle_path, db_path, out_path):
    print(f"\nLoading bundle from: {bundle_path}")
    engine = load_scenario_model_bundle(bundle_path)
    print(f"Bundle loaded. Targets: {list(engine.target_names)}")

    rows, macro = load_market_rows(db_path)

    # Lower minimum history since we only have 250 rows
    # Need at least 50 rows of history and 21 rows forward
    min_history = 50
    results = []
    skipped = 0

    total = len(rows) - HORIZON - min_history
    print(f"Running backtest over {total} dates (min_history={min_history}, horizon={HORIZON})...\n")

    for idx in range(min_history, len(rows) - HORIZON):
        row_date = rows[idx]["date"]

        state = build_state_vector(rows, macro, idx, engine.state_feature_names)

        try:
            result = engine.predict(
                NEUTRAL_SHOCKS,
                current_state=state,
                state_sample_date=row_date,
                out_of_range_policy="warn",
            )
        except Exception as exc:
            print(f"  Skipped {row_date}: {exc}")
            skipped += 1
            continue

        for target in engine.target_names:
            pred = result["targets"][target]["baseline_prediction"]
            actual = compute_actual_return(rows, idx, target)
            if actual is None:
                continue

            error = pred - actual
            results.append({
                "date":             row_date,
                "target":           target,
                "unit":             result["targets"][target]["unit"],
                "prediction":       round(pred, 4),
                "actual":           round(actual, 4),
                "error":            round(error, 4),
                "abs_error":        round(abs(error), 4),
                "squared_error":    round(error ** 2, 6),
                "direction_correct": int((pred >= 0) == (actual >= 0)),
                "confidence":       result["targets"][target]["confidence_level"],
            })

        if (idx - min_history) % 10 == 0:
            pct_done = (idx - min_history + 1) / total * 100
            print(f"  {pct_done:.0f}% — {row_date}")

    print(f"\nDone. {len(results)} predictions, {skipped} skipped.")

    if not results:
        print("No results — check DB has data and bundle is correct.")
        return

    # Write CSV
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date","target","unit","prediction","actual","error",
                  "abs_error","squared_error","direction_correct","confidence"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {out_path}\n")

    # Summary table
    print(f"{'Target':<30} {'N':>5} {'MAE':>8} {'RMSE':>8} {'Dir%':>6}")
    print("-" * 60)
    for target in engine.target_names:
        trows = [r for r in results if r["target"] == target]
        if not trows:
            continue
        n    = len(trows)
        mae  = statistics.mean(r["abs_error"] for r in trows)
        rmse = math.sqrt(statistics.mean(r["squared_error"] for r in trows))
        dacc = sum(r["direction_correct"] for r in trows) / n * 100
        unit = trows[0]["unit"]
        print(f"{target:<30} {n:>5} {mae:>7.3f} {rmse:>7.3f} {dacc:>5.1f}%  [{unit}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="scenario_modeling/artifacts/scenario-playground-ridge-v1.0.0/v1.0.0")
    parser.add_argument("--db",     default="backend/insightpulse.db")
    parser.add_argument("--out",    default="scenario_modeling/backtest_results.csv")
    args = parser.parse_args()
    run_backtest(args.bundle, args.db, args.out)