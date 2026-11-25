"""
run_confidence_validation.py   (Task 6, v1)

Validates the confidence labels and (optionally) prediction intervals of the
Ridge scenario model.

STATUS: v1. Calibrates and tests intervals on Khalil's saved out-of-sample
walk-forward predictions (2021 to Nov 2023, trained through Nov 6 2023).
The 2025-2026 backtest window is NOT evaluated: part of it overlaps the
Ridge team's locked final-test set. Pass --backtest-window-approved only after
Khalil agrees.

SCOPE WARNING
The backtest runs the model with all eight shocks at zero. It therefore checks
the BASELINE forecast (predicted 21-trading-day market move) against what
actually happened. It cannot check the INCREMENTAL scenario effect, because
there is no real counterfactual. Any interval produced here describes
baseline forecast error only. Never present it as uncertainty about a
scenario effect.

PREREQUISITES (see plan, Step 0)
  1. Rerun Task 2 with corrected inputs (macro table, CPI/GDP features,
     MA200 history, XLU for Utilities).
  2. Extend run_backtest.py to save the diagnostics returned by
     extract_confidence_fields() below, and pass diagnostic_as_of_date=row_date
     to engine.predict(). Otherwise freshness is measured against today and
     every historical row looks stale.

USAGE (from repo root)
    python scenario_modeling/run_confidence_validation.py --selftest
    python scenario_modeling/run_confidence_validation.py `
        --walk-forward scenario_modeling/ridge_walk_forward_predictions_2021_2023.csv `
        --db backend/insightpulse.db `
        --out scenario_modeling/confidence_validation_v1.json
"""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# These values are PROPOSALS. Agree on them with the team BEFORE running on
# real data, then do not change them after looking at test results.
# ---------------------------------------------------------------------------
CONFIG = {
    "horizon_days": 21,
    "nominal_coverage": 0.80,      # target share of outcomes inside the interval
    "calibration_fraction": 0.50,  # earliest share of dates used to set widths
    "embargo_calendar_days": 30,   # about 21 trading days, so outcome windows do not overlap
    "min_test_dates": 60,
    "coverage_tolerance": 0.10,    # pass if coverage is within +/- 10 points
    "bootstrap_iters": 1000,
    "block_length": None,          # None = derive from date spacing (windows overlap)
    "seed": 42,
}

TARGETS = [
    "sp500", "nasdaq", "ten_year_yield_bps", "dxy", "gold", "oil",
    "technology", "energy", "financials", "utilities",
    "healthcare", "consumer_discretionary",
]


# ---------------------------------------------------------------------------
# 1. Loading and splitting   (implemented)
# ---------------------------------------------------------------------------
def load_backtest(path):
    """Read backtest_results.csv. Numeric columns become floats."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["date"] = r["date"][:10]
            r["prediction"] = float(r["prediction"])
            r["actual"] = float(r["actual"])
            rows.append(r)
    return rows


def split_dates(dates, calibration_fraction, embargo_calendar_days):
    """
    Chronological split: calibration | embargo | test. Test starts only after
    the embargo (in calendar days) so each calibration outcome, known about 21
    trading days after its date, is already known when the first test date arrives.
    """
    from datetime import date, timedelta
    dates = sorted(set(dates))
    c = int(len(dates) * calibration_fraction)
    cal = dates[:c]
    cutoff = (date.fromisoformat(cal[-1]) + timedelta(days=embargo_calendar_days)).isoformat()
    return {
        "calibration": cal,
        "embargo": [d for d in dates[c:] if d <= cutoff],
        "test": [d for d in dates[c:] if d > cutoff],
    }


# ---------------------------------------------------------------------------
# 2. Interval helpers   (implemented)
# ---------------------------------------------------------------------------
def conformal_halfwidth(abs_errors, coverage):
    """
    Split conformal half-width: the ceil((n+1)*coverage)-th smallest absolute
    error from the calibration period. Interval = prediction +/- half-width.
    Returns inf if there is too little calibration data for this coverage.
    """
    n = len(abs_errors)
    k = math.ceil((n + 1) * coverage)
    if n == 0 or k > n:
        return math.inf
    return sorted(abs_errors)[k - 1]


def quantile(values, q):
    """Linear-interpolation quantile, q in [0, 1]."""
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def unconditional_interval(calibration_actuals, coverage):
    """
    Baseline interval: the middle `coverage` share of past outcomes.
    Says only "the market usually moves this much". A model interval should
    be narrower than this to be worth showing.
    """
    tail = (1 - coverage) / 2
    return quantile(calibration_actuals, tail), quantile(calibration_actuals, 1 - tail)


def block_bootstrap_ci(values, block_length, iters, seed, stat=statistics.mean):
    """
    95% CI for a statistic of a time-ordered list, resampling consecutive
    blocks. Use blocks (not single points) because 21-day windows overlap.
    """
    n = len(values)
    if n == 0:
        return (math.nan, math.nan)
    block_length = max(1, min(block_length, n))
    n_blocks = math.ceil(n / block_length)
    rng = random.Random(seed)
    stats = []
    for _ in range(iters):
        sample = []
        for _ in range(n_blocks):
            start = rng.randint(0, n - block_length)
            sample.extend(values[start:start + block_length])
        stats.append(stat(sample[:n]))
    stats.sort()
    return stats[int(0.025 * iters)], stats[min(iters - 1, int(0.975 * iters))]


# ---------------------------------------------------------------------------
# 3. Interval evaluation for one target   (implemented, glue only)
# ---------------------------------------------------------------------------
def derive_block_length(dates, window_days=30):
    """Number of consecutive dates whose 21-trading-day windows overlap."""
    from datetime import date
    ds = sorted(set(dates))
    gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(ds, ds[1:])]
    med = statistics.median(gaps) if gaps else window_days
    return max(2, math.ceil(window_days / max(med, 1)))


def evaluate_intervals_for_target(target_rows, cfg):
    """
    target_rows: rows for ONE target. Calibrates on the early dates and
    measures on the later dates only. Also compares the model to a constant
    forecast (calibration median) to show whether the model adds anything.
    """
    by_date = {r["date"]: r for r in target_rows}
    split = split_dates(by_date, cfg["calibration_fraction"], cfg["embargo_calendar_days"])
    calib = [by_date[d] for d in split["calibration"]]
    test = [by_date[d] for d in split["test"]]

    cal_abs = [abs(r["prediction"] - r["actual"]) for r in calib]
    hw = conformal_halfwidth(cal_abs, cfg["nominal_coverage"])
    hits = [1 if abs(r["prediction"] - r["actual"]) <= hw else 0 for r in test]

    lo, hi = unconditional_interval([r["actual"] for r in calib], cfg["nominal_coverage"])
    base_hits = [1 if lo <= r["actual"] <= hi else 0 for r in test]

    block = cfg["block_length"] or derive_block_length(by_date)
    cov_lo, cov_hi = block_bootstrap_ci(hits, block, cfg["bootstrap_iters"], cfg["seed"])

    const = statistics.median([r["actual"] for r in calib])
    mae_model = statistics.mean(abs(r["prediction"] - r["actual"]) for r in test) if test else math.nan
    mae_const = statistics.mean(abs(const - r["actual"]) for r in test) if test else math.nan
    return {
        "n_calibration": len(calib),
        "n_test": len(test),
        "calibration_range": [split["calibration"][0], split["calibration"][-1]] if calib else None,
        "test_range": [split["test"][0], split["test"][-1]] if test else None,
        "block_length_dates": block,
        "model_width": 2 * hw,
        "baseline_width": hi - lo,
        "model_coverage": statistics.mean(hits) if hits else math.nan,
        "model_coverage_ci": [cov_lo, cov_hi],
        "baseline_coverage": statistics.mean(base_hits) if base_hits else math.nan,
        "test_mae_model": mae_model,
        "test_mae_constant_forecast": mae_const,
        "model_beats_constant": bool(mae_model < mae_const),
        "coverage_by_vix_regime": None,   # not testable: no VIX history for 2021-2023 in the DB
    }


# ---------------------------------------------------------------------------
# 4. Display policy   (implemented, encodes the PROPOSED rules)
# ---------------------------------------------------------------------------
def decide_display_policy(metrics, cfg, inputs_fixed):
    """
    Returns {"policy": "numeric_interval" | "qualitative_only", "reasons": [...]}.
    Default is qualitative_only. A target must pass every rule to get a number.
    """
    reasons = []
    if not inputs_fixed:
        reasons.append("inputs not confirmed as corrected")
    if metrics["n_test"] < cfg["min_test_dates"]:
        reasons.append(f"only {metrics['n_test']} test dates (need {cfg['min_test_dates']})")
    if not math.isfinite(metrics["model_width"]):
        reasons.append("too little calibration data for this coverage level")
    else:
        gap = abs(metrics["model_coverage"] - cfg["nominal_coverage"])
        if gap > cfg["coverage_tolerance"]:
            reasons.append(f"coverage {metrics['model_coverage']:.2f} is outside the tolerance")
        if metrics["model_width"] >= metrics["baseline_width"]:
            reasons.append("interval is not narrower than the unconditional baseline")
    if not metrics["model_beats_constant"]:
        reasons.append("point forecast is not more accurate than a constant forecast")
    if metrics["coverage_by_vix_regime"] is None:
        reasons.append("coverage not checked within VIX regimes (no VIX history for this period)")
    return {
        "policy": "qualitative_only" if reasons else "numeric_interval",
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 5. Pulling diagnostics out of engine.predict()   (to paste into run_backtest.py)
# ---------------------------------------------------------------------------
def extract_confidence_fields(result, target):
    """
    Fields to save per (date, target) in the corrected backtest. Keys follow
    scenario_runtime.py. Test once against a real engine.predict() result.
    Call engine.predict(..., diagnostic_as_of_date=row_date).
    """
    conf = result["confidence"]
    t = conf["targets"][target]
    joint = conf["joint_feature_distance"]["rows"]["baseline"]
    return {
        "confidence_level": result["targets"][target]["confidence_level"],
        "overall_research_confidence": t["overall_research_confidence"],
        "distance_status": joint["distance_status"],
        "distance_percentile": joint["distance_percentile_vs_training_neighbors"],
        "freshness_status": conf["state_freshness"]["freshness_status"],
        "vix_regime": t["current_vix_regime"],
        "regime_error_ratio": t["similar_regime_error_ratio"],
        "n_input_range_warnings": t["input_range_warning_count"],
        "n_state_extrapolation_warnings": t["state_extrapolation_warning_count"],
        "n_model_performance_warnings": t["model_performance_warning_count"],
    }


# ---------------------------------------------------------------------------
# 6. TODO sections   (not implemented)
# ---------------------------------------------------------------------------
def error_by_label(rows, label_field):
    """
    TODO (Step 3): for each value of `label_field` (distance_status,
    freshness_status, overall_research_confidence, vix_regime), report N, MAE
    and a block-bootstrap CI per target. Question: do the "worse" labels
    really have larger errors? A label that does not separate good from bad
    cases should be reworded or dropped.
    Note: 'confidence_level' is fixed per target, so compare it ACROSS targets
    (for example MAE relative to the best naive baseline from Task 3).
    """
    raise NotImplementedError


def audit_probabilities(db_path):
    """
    The only probability the app shows is mood_5d.probability. It is a vote
    share of 5 indicators (see market_state_service._build_mood_5d). This
    recomputes it on each past date with the same rule and checks it against
    the S&P 500 5-trading-day forward direction. Reads market_data only.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT date, sp500, nasdaq, vix, gold, tlt FROM market_data "
                        "ORDER BY date").fetchall()
    conn.close()
    rows = [r for r in rows if all(v is not None for v in r)]
    out = []
    for i in range(5, len(rows) - 5):
        up = dn = 0
        for k, invert in ((1, False), (2, False), (3, True), (4, False), (5, False)):
            v = rows[i][k] / rows[i - 5][k] - 1
            if v == 0:
                continue
            bullish = (v > 0) != invert
            up += bullish
            dn += not bullish
        if up > dn:
            lab, p = "bullish", up / (up + dn)
        elif dn > up:
            lab, p = "bearish", dn / (up + dn)
        else:
            lab, p = "neutral", 0.5
        fwd_up = int(rows[i + 5][1] > rows[i][1])
        p_up = p if lab == "bullish" else (1 - p if lab == "bearish" else 0.5)
        out.append((rows[i][0], lab, round(p, 2), p_up, fwd_up))
    n = len(out)
    base = statistics.mean(o[4] for o in out)
    table = defaultdict(list)
    for _, lab, p, _, f in out:
        table[(lab, p)].append(f)
    return {
        "what_is_shown": "mood_5d.probability, displayed as 'confidence N%'",
        "how_computed": "share of 5 indicators (S&P, Nasdaq, VIX, gold, TLT 5-day returns) agreeing with the label",
        "n_dates": n, "date_range": [out[0][0], out[-1][0]],
        "windows_overlap": True,
        "base_rate_sp500_up_next_5d": base,
        "brier_displayed": statistics.mean((o[3] - o[4]) ** 2 for o in out),
        "brier_base_rate": statistics.mean((base - o[4]) ** 2 for o in out),
        "brier_coin_flip": statistics.mean((0.5 - o[4]) ** 2 for o in out),
        "label_direction_hit_rate": statistics.mean(
            int((o[1] == "bullish") == bool(o[4])) for o in out if o[1] != "neutral"),
        "reliability": [{"label": k[0], "shown_probability": k[1], "n": len(v),
                         "observed_up_frequency": statistics.mean(v)}
                        for k, v in sorted(table.items())],
        "verdict": "Not a calibrated probability. Do not display as a percentage.",
    }


def asymmetric_interval_variant(target_rows, cfg):
    """
    TODO (optional, Step 4): interval from signed errors, because the model
    may lean positive (Task 4 pattern). Compare against the symmetric version.
    Decide before running whether this variant replaces or only supplements it.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 7. Output   (implemented)
# ---------------------------------------------------------------------------
def write_artifact(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Self test: synthetic data only, never touches real results
# ---------------------------------------------------------------------------
def selftest():
    # Conformal quantile arithmetic
    assert conformal_halfwidth([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.8) == 8   # ceil(10*0.8)=8
    assert conformal_halfwidth([1, 2, 3], 0.8) == math.inf              # too few points
    assert quantile([0, 10], 0.5) == 5

    # Split keeps the gap and order
    s = split_dates([f"2026-01-{d:02d}" for d in range(1, 29)], 0.5, 3)
    assert s["calibration"][-1] < s["embargo"][0] < s["test"][0]
    assert len(s["embargo"]) == 3

    # iid Laplace errors: split conformal coverage should land near 80%
    rng = random.Random(1)
    rows = []
    for i in range(600):
        pred = rng.gauss(0, 2)
        actual = pred + (rng.expovariate(1) - rng.expovariate(1))
        rows.append({"date": (__import__("datetime").date(2020, 1, 1) + __import__("datetime").timedelta(days=9 * i)).isoformat(), "prediction": pred, "actual": actual})
    m = evaluate_intervals_for_target(rows, CONFIG)
    assert abs(m["model_coverage"] - 0.80) < 0.07, m
    assert m["model_coverage_ci"][0] <= 0.80 <= m["model_coverage_ci"][1], m
    print("selftest passed:",
          f"coverage={m['model_coverage']:.3f}",
          f"ci=[{m['model_coverage_ci'][0]:.3f}, {m['model_coverage_ci'][1]:.3f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Task 6 confidence validation v1")
    p.add_argument("--selftest", action="store_true", help="run checks on synthetic data")
    p.add_argument("--walk-forward", default="scenario_modeling/ridge_walk_forward_predictions_2021_2023.csv")
    p.add_argument("--db", default="backend/insightpulse.db")
    p.add_argument("--out", default="scenario_modeling/confidence_validation_v1.json")
    p.add_argument("--backtest-window-approved", action="store_true",
                   help="only after Khalil agrees: the 2025-26 window overlaps the locked final-test set")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    if args.backtest_window_approved:
        raise SystemExit("Backtest-window evaluation is not implemented in v1. Add it only after the "
                         "locked final-test overlap is agreed with the Ridge team.")

    rows = load_backtest(args.walk_forward)
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)

    payload = {
        "artifact_version": "confidence_validation_v1",
        "applies_to": "baseline forecast only (not the incremental scenario effect)",
        "causal_effect_claimed": False,
        "interval_data": "Ridge out-of-sample walk-forward predictions, 2021-01 to 2023-11 (model trained through 2023-11-06)",
        "backtest_window_check": "NOT RUN: part of the 2025-26 backtest window overlaps the locked final-test set; pending agreement with the Ridge team",
        "config": CONFIG,
        "label_findings": {
            "scenario_confidence_ui": "Always 'Low'; never returns High. Describes input familiarity, not accuracy.",
            "research_confidence_per_target": "Constant over time for every target, so it cannot be tested against error within a target. Across targets, volatility-scaled MAE does not rank by label.",
        },
        "targets": {},
    }
    for target in TARGETS:
        if target not in by_target:
            continue
        metrics = evaluate_intervals_for_target(by_target[target], CONFIG)
        payload["targets"][target] = {
            "interval_metrics": metrics,
            "display": decide_display_policy(metrics, CONFIG, True),
        }
        m = metrics
        print(f"{target:<24} cov={m['model_coverage']:.2f} (base {m['baseline_coverage']:.2f})  "
              f"width={m['model_width']:.1f} vs {m['baseline_width']:.1f}  "
              f"beats_const={m['model_beats_constant']}  -> "
              f"{payload['targets'][target]['display']['policy']}")
    payload["probability_audit"] = audit_probabilities(args.db)
    pa = payload["probability_audit"]
    print(f"\nMood probability: Brier {pa['brier_displayed']:.3f} vs coin flip {pa['brier_coin_flip']:.3f}, "
          f"direction hit {pa['label_direction_hit_rate']:.2f} over {pa['n_dates']} dates")
    write_artifact(args.out, payload)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
