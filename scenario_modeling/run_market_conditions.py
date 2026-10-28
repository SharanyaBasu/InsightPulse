"""
run_market_conditions.py
------------------------
Task 4: Test performance across market conditions.

Splits backtest results into four regime dimensions:
  1. Rising vs falling market  (SP500 21-day actual return)
  2. High vs low volatility    (VIX level from market_data DB)

For each regime also reports:
  - 95% confidence interval on MAE using bootstrap
  - Whether performance concentrates in one favorable sub-period

Usage (from repo root):
    python scenario_modeling/run_market_conditions.py \
        --results scenario_modeling/backtest_results.csv \
        --db      backend/insightpulse.db \
        --out     scenario_modeling/market_conditions_results.csv
"""

import argparse
import csv
import math
import random
import sqlite3
import statistics
from pathlib import Path

TARGETS = [
    "sp500", "nasdaq", "ten_year_yield_bps", "dxy", "gold", "oil",
    "technology", "energy", "financials", "utilities",
    "healthcare", "consumer_discretionary",
]

BOOTSTRAP_ITERS = 1000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def mae_rmse_dir(subset):
    if not subset:
        return None
    n    = len(subset)
    errs = [abs(float(r["error"])) for r in subset]
    sq   = [float(r["squared_error"]) for r in subset]
    dirs = [int(r["direction_correct"]) for r in subset]
    mae  = statistics.mean(errs)
    rmse = math.sqrt(statistics.mean(sq))
    dacc = sum(dirs) / n * 100
    return {"n": n, "mae": mae, "rmse": rmse, "dir": dacc, "errs": errs}


def bootstrap_mae_ci(errs, iters=BOOTSTRAP_ITERS, seed=RANDOM_SEED):
    """95% CI on MAE via bootstrap resampling."""
    rng = random.Random(seed)
    n = len(errs)
    samples = []
    for _ in range(iters):
        resample = [rng.choice(errs) for _ in range(n)]
        samples.append(statistics.mean(resample))
    samples.sort()
    lo = samples[int(0.025 * iters)]
    hi = samples[int(0.975 * iters)]
    return lo, hi


def half_yearly_mae(subset, label):
    """
    Split subset into two halves chronologically and compare MAE.
    Flags if one half has MAE more than 50% lower than the other.
    """
    if len(subset) < 10:
        return "insufficient data for period analysis"
    dated = sorted(subset, key=lambda r: r["date"])
    mid   = len(dated) // 2
    first_half  = dated[:mid]
    second_half = dated[mid:]
    mae1 = statistics.mean(abs(float(r["error"])) for r in first_half)
    mae2 = statistics.mean(abs(float(r["error"])) for r in second_half)
    ratio = max(mae1, mae2) / min(mae1, mae2) if min(mae1, mae2) > 0 else float("inf")
    period1 = f"{first_half[0]['date']} to {first_half[-1]['date']}"
    period2 = f"{second_half[0]['date']} to {second_half[-1]['date']}"
    note = f"MAE {period1}: {mae1:.2f}  |  MAE {period2}: {mae2:.2f}"
    if ratio > 1.5:
        better = "first" if mae1 < mae2 else "second"
        return f"WARNING: {better} half has 50%+ lower MAE — performance may depend on one period. {note}"
    return f"No strong period concentration. {note}"


# ---------------------------------------------------------------------------
# Regime loaders
# ---------------------------------------------------------------------------

def load_vix(db_path):
    """Load date -> VIX level from market_data table."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT date, vix FROM market_data WHERE vix IS NOT NULL ORDER BY date")
    result = {row[0]: float(row[1]) for row in cur.fetchall()}
    conn.close()
    return result


def build_regimes(rows, vix_map):
    """
    Returns a dict: date -> {market: rising/falling, volatility: high/low}
    Market: based on SP500 21-day actual return Q1/Q3 split
    Volatility: based on VIX median split at backtest dates
    """
    sp500_ret = {r["date"]: float(r["actual"]) for r in rows if r["target"] == "sp500"}
    sp_vals   = sorted(sp500_ret.values())
    q1 = sp_vals[len(sp_vals) // 4]
    q3 = sp_vals[3 * len(sp_vals) // 4]

    # VIX at each backtest date (use closest available date if exact not found)
    vix_dates = sorted(vix_map.keys())
    def nearest_vix(d):
        for vd in reversed(vix_dates):
            if vd <= d:
                return vix_map[vd]
        return None

    vix_at_dates = {d: nearest_vix(d) for d in sp500_ret}
    vix_vals = [v for v in vix_at_dates.values() if v is not None]
    vix_median = statistics.median(vix_vals) if vix_vals else 20.0

    regimes = {}
    for d, sp_ret in sp500_ret.items():
        if sp_ret <= q1:
            market = "stress"
        elif sp_ret >= q3:
            market = "rally"
        else:
            market = "neutral"

        vix_val = vix_at_dates.get(d)
        volatility = "high_vol" if (vix_val is not None and vix_val >= vix_median) else "low_vol"

        regimes[d] = {"market": market, "volatility": volatility}

    return regimes, q1, q3, vix_median


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(results_path, db_path, out_path):
    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))

    vix_map = load_vix(db_path)
    regimes, q1, q3, vix_median = build_regimes(rows, vix_map)

    sp500_ret = {r["date"]: float(r["actual"]) for r in rows if r["target"] == "sp500"}
    vix_at_dates = {}
    vix_dates = sorted(vix_map.keys())
    for d in sp500_ret:
        for vd in reversed(vix_dates):
            if vd <= d:
                vix_at_dates[d] = vix_map[vd]
                break

    n_total   = len(sp500_ret)
    stress_n  = sum(1 for r in regimes.values() if r["market"] == "stress")
    neutral_n = sum(1 for r in regimes.values() if r["market"] == "neutral")
    rally_n   = sum(1 for r in regimes.values() if r["market"] == "rally")
    high_n    = sum(1 for r in regimes.values() if r["volatility"] == "high_vol")
    low_n     = sum(1 for r in regimes.values() if r["volatility"] == "low_vol")

    print("=" * 72)
    print("Task 4: Performance Across Market Conditions")
    print("=" * 72)
    print(f"\nRegime definitions (N = {n_total} dates):")
    print(f"  Market direction  — SP500 21-day forward return:")
    print(f"    Stress  (bottom 25%): SP500 <= {q1:.2f}%   [{stress_n} dates]")
    print(f"    Neutral (middle 50%): {q1:.2f}% to {q3:.2f}%   [{neutral_n} dates]")
    print(f"    Rally   (top 25%):    SP500 >= {q3:.2f}%   [{rally_n} dates]")
    print(f"  Volatility — VIX at prediction date (median = {vix_median:.1f}):")
    print(f"    High vol: VIX >= {vix_median:.1f}   [{high_n} dates]")
    print(f"    Low vol:  VIX <  {vix_median:.1f}   [{low_n} dates]")

    csv_rows = []

    # --- DIMENSION 1: Market direction (stress / neutral / rally) ---
    print("\n" + "=" * 72)
    print("DIMENSION 1: Rising vs Falling Market")
    print("=" * 72)
    print(f"\n{'Target':<30} {'Regime':<10} {'N':>4} {'MAE':>7} {'95% CI':>16} {'Dir%':>6}")
    print("-" * 72)

    for target in TARGETS:
        trows = [r for r in rows if r["target"] == target]
        unit  = "bps" if "bps" in trows[0]["unit"] else "%"
        for regime in ["stress", "neutral", "rally"]:
            subset = [r for r in trows if regimes.get(r["date"], {}).get("market") == regime]
            s = mae_rmse_dir(subset)
            if not s:
                continue
            lo, hi = bootstrap_mae_ci(s["errs"])
            print(f"{target:<30} {regime:<10} {s['n']:>4} "
                  f"{s['mae']:>5.2f}{unit} "
                  f"[{lo:.2f},{hi:.2f}]{unit} "
                  f"{s['dir']:>5.1f}%")
            csv_rows.append({
                "dimension": "market_direction", "regime": regime,
                "target": target, "unit": trows[0]["unit"],
                "n": s["n"], "mae": round(s["mae"], 3),
                "rmse": round(s["rmse"], 3), "dir_pct": round(s["dir"], 1),
                "mae_ci_lo": round(lo, 3), "mae_ci_hi": round(hi, 3),
            })
        print()

    # --- DIMENSION 2: Volatility (high / low VIX) ---
    print("=" * 72)
    print("DIMENSION 2: High vs Low Volatility (VIX)")
    print("=" * 72)
    print(f"\n{'Target':<30} {'Regime':<10} {'N':>4} {'MAE':>7} {'95% CI':>16} {'Dir%':>6}")
    print("-" * 72)

    for target in TARGETS:
        trows = [r for r in rows if r["target"] == target]
        unit  = "bps" if "bps" in trows[0]["unit"] else "%"
        for regime in ["high_vol", "low_vol"]:
            subset = [r for r in trows if regimes.get(r["date"], {}).get("volatility") == regime]
            s = mae_rmse_dir(subset)
            if not s:
                continue
            lo, hi = bootstrap_mae_ci(s["errs"])
            print(f"{target:<30} {regime:<10} {s['n']:>4} "
                  f"{s['mae']:>5.2f}{unit} "
                  f"[{lo:.2f},{hi:.2f}]{unit} "
                  f"{s['dir']:>5.1f}%")
            csv_rows.append({
                "dimension": "volatility", "regime": regime,
                "target": target, "unit": trows[0]["unit"],
                "n": s["n"], "mae": round(s["mae"], 3),
                "rmse": round(s["rmse"], 3), "dir_pct": round(s["dir"], 1),
                "mae_ci_lo": round(lo, 3), "mae_ci_hi": round(hi, 3),
            })
        print()

    # --- DIMENSION 3: Period concentration check ---
    print("=" * 72)
    print("DIMENSION 3: Period Concentration — does one sub-period drive results?")
    print("=" * 72)
    print()
    for target in TARGETS:
        trows = [r for r in rows if r["target"] == target]
        note  = half_yearly_mae(trows, target)
        print(f"  {target:<30} {note}")

    # Write CSV
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dimension","regime","target","unit","n","mae","rmse",
                  "dir_pct","mae_ci_lo","mae_ci_hi"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="scenario_modeling/backtest_results.csv")
    parser.add_argument("--db",      default="backend/insightpulse.db")
    parser.add_argument("--out",     default="scenario_modeling/market_conditions_results.csv")
    args = parser.parse_args()
    run(args.results, args.db, args.out)