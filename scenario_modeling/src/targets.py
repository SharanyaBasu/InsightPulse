"""Construct the 12 Scenario Playground 21-session market targets.

Step 5 consumes the selected Step 4 sampling index and accepted Step 3 target
caches. It does not construct shocks, state features, splits, or models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .cache import write_text_atomic
from .config import INTERIM_ROOT, PROJECT_ROOT, RAW_ROOT
from .sampling import (
    TARGET_HORIZON_SESSIONS,
    TARGET_SPECS,
    TargetSpec,
    load_canonical_calendar,
)


getcontext().prec = 32

SELECTED_SAMPLING_PATH = (
    INTERIM_ROOT / "sampling" / "selected_sampling_index.csv"
)
TARGET_OUTPUT_ROOT = INTERIM_ROOT / "targets"

TARGET_NAMES = tuple(spec.target for spec in TARGET_SPECS)
PRICE_TARGET_NAMES = tuple(
    name for name in TARGET_NAMES if name != "ten_year_yield_bps"
)

TARGET_KEY_COLUMNS = (
    "sample_date",
    "model_availability_date",
    "state_cutoff_date",
    "target_baseline_date",
    "target_end_date",
    "target_horizon_sessions",
    "event_types_json",
    "event_ids_json",
)
TARGET_OUTPUT_COLUMNS = TARGET_KEY_COLUMNS + TARGET_NAMES

# Review thresholds identify observations that need human inspection. They do
# not clip, delete, or relabel a valid cached observation.
EXTREME_REVIEW_THRESHOLDS = {
    **{target: 50.0 for target in PRICE_TARGET_NAMES},
    "oil": 150.0,
    "ten_year_yield_bps": 100.0,
}


@dataclass(frozen=True)
class LoadedTargetSeries:
    spec: TargetSpec
    values: pd.Series
    path: Path


def _as_iso_date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_to_csv_text(frame: pd.DataFrame) -> str:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    return buffer.getvalue()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_selected_sampling_index(
    path: Path = SELECTED_SAMPLING_PATH,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Selected Step 4 sampling index is missing: {path}"
        )
    frame = pd.read_csv(path)
    required = set(TARGET_KEY_COLUMNS) | {
        "sampling_design",
        "label_eligible",
        "has_21_future_sessions",
        "full_horizon_within_shared_coverage",
        "all_target_values_available",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Selected sampling index lacks columns: {missing}")
    if frame.empty:
        raise ValueError("Selected sampling index contains no rows.")
    if frame["sample_date"].duplicated().any():
        raise ValueError("Selected sampling index has duplicate sample dates.")
    if not (frame["sampling_design"] == "EVENT_AWARE").all():
        raise ValueError("Step 5 expects the selected EVENT_AWARE design.")
    for flag in (
        "label_eligible",
        "has_21_future_sessions",
        "full_horizon_within_shared_coverage",
        "all_target_values_available",
    ):
        if not frame[flag].astype(bool).all():
            raise ValueError(f"Selected sampling index contains false {flag}.")
    if not (
        pd.to_numeric(frame["target_horizon_sessions"], errors="coerce")
        == TARGET_HORIZON_SESSIONS
    ).all():
        raise ValueError("Selected rows do not all use the 21-session horizon.")
    for column in (
        "sample_date",
        "model_availability_date",
        "state_cutoff_date",
        "target_baseline_date",
        "target_end_date",
    ):
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Selected index has invalid dates in {column}.")
        frame[column] = parsed.dt.strftime("%Y-%m-%d")
    if not (
        frame["state_cutoff_date"] == frame["target_baseline_date"]
    ).all():
        raise ValueError(
            "Target baseline must remain the Step 4 pre-shock state cutoff."
        )
    if not (
        pd.to_datetime(frame["target_end_date"])
        > pd.to_datetime(frame["target_baseline_date"])
    ).all():
        raise ValueError("Every target end must follow its baseline.")
    return frame.sort_values("sample_date").reset_index(drop=True)


def _load_yahoo_series(spec: TargetSpec, path: Path) -> pd.Series:
    frame = pd.read_csv(path, usecols=["Date", spec.value_field])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[spec.value_field] = pd.to_numeric(
        frame[spec.value_field], errors="coerce"
    )
    frame = frame.dropna(subset=["Date"])
    if frame["Date"].duplicated().any():
        raise ValueError(f"Duplicate market dates in {path}")
    values = frame.set_index("Date")[spec.value_field].sort_index()
    values.index = values.index.strftime("%Y-%m-%d")
    return values


def _load_fred_series(spec: TargetSpec, path: Path) -> pd.Series:
    payload = _read_json(path)
    frame = pd.DataFrame(payload.get("observations", []))
    if not {"date", "value"}.issubset(frame.columns):
        raise ValueError(f"FRED observations are malformed in {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate FRED dates in {path}")
    values = frame.set_index("date")["value"].sort_index()
    values.index = values.index.strftime("%Y-%m-%d")
    return values


def load_target_series(
    raw_root: Path = RAW_ROOT,
) -> tuple[dict[str, LoadedTargetSeries], list[Path]]:
    loaded: dict[str, LoadedTargetSeries] = {}
    paths: list[Path] = []
    for spec in TARGET_SPECS:
        path = raw_root / spec.relative_path
        if not path.exists():
            raise FileNotFoundError(f"Selected target cache is missing: {path}")
        if spec.file_format == "fred_json":
            values = _load_fred_series(spec, path)
        else:
            values = _load_yahoo_series(spec, path)
        if values.dropna().empty:
            raise ValueError(f"No usable {spec.value_field} values in {path}")
        loaded[spec.target] = LoadedTargetSeries(spec, values, path)
        paths.append(path)
    return loaded, paths


def calculate_price_return_pct(
    baseline: pd.Series, future: pd.Series
) -> pd.Series:
    baseline_values = pd.to_numeric(baseline, errors="coerce")
    future_values = pd.to_numeric(future, errors="coerce")
    if baseline_values.isna().any() or future_values.isna().any():
        raise ValueError("Price-return endpoints contain missing values.")
    if (baseline_values == 0).any():
        raise ValueError("A percentage-return baseline is zero.")
    result = 100.0 * (future_values / baseline_values - 1.0)
    if not result.map(math.isfinite).all():
        raise ValueError("Price-return calculation produced non-finite values.")
    return result


def calculate_yield_change_bps(
    baseline: pd.Series, future: pd.Series
) -> pd.Series:
    baseline_values = pd.to_numeric(baseline, errors="coerce")
    future_values = pd.to_numeric(future, errors="coerce")
    if baseline_values.isna().any() or future_values.isna().any():
        raise ValueError("Yield endpoints contain missing values.")
    result = 100.0 * (future_values - baseline_values)
    if not result.map(math.isfinite).all():
        raise ValueError("Yield calculation produced non-finite values.")
    return result


def _exact_values(
    series: pd.Series,
    dates: pd.Series,
    *,
    target_name: str,
    endpoint_name: str,
) -> pd.Series:
    requested = [_as_iso_date(value) for value in dates]
    values = series.reindex(requested)
    values.index = dates.index
    missing_positions = values[values.isna()].index.tolist()
    if missing_positions:
        examples = [
            {
                "row": int(position),
                "date": requested[int(position)],
            }
            for position in missing_positions[:10]
        ]
        raise ValueError(
            f"{target_name} lacks exact {endpoint_name} values: {examples}"
        )
    return values.astype(float)


def _target_values(
    target_name: str,
    loaded: LoadedTargetSeries,
    sampling: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    baseline = _exact_values(
        loaded.values,
        sampling["target_baseline_date"],
        target_name=target_name,
        endpoint_name="baseline",
    )
    future = _exact_values(
        loaded.values,
        sampling["target_end_date"],
        target_name=target_name,
        endpoint_name="future",
    )
    if target_name == "ten_year_yield_bps":
        target = calculate_yield_change_bps(baseline, future)
    else:
        target = calculate_price_return_pct(baseline, future)
    return baseline, future, target


def _validate_exact_horizons(
    sampling: pd.DataFrame,
    raw_root: Path,
) -> dict[str, Any]:
    sessions = load_canonical_calendar(raw_root)
    positions = {
        session.isoformat(): position for position, session in enumerate(sessions)
    }
    gaps: list[int] = []
    for row in sampling.itertuples(index=False):
        baseline = str(row.target_baseline_date)
        target_end = str(row.target_end_date)
        if baseline not in positions or target_end not in positions:
            raise ValueError(
                f"Target dates are not canonical sessions: {baseline}, {target_end}"
            )
        gaps.append(positions[target_end] - positions[baseline])
    invalid = [gap for gap in gaps if gap != TARGET_HORIZON_SESSIONS]
    if invalid:
        raise ValueError(f"Found non-21-session target gaps: {invalid[:10]}")
    return {
        "row_count": len(gaps),
        "minimum_session_gap": min(gaps),
        "maximum_session_gap": max(gaps),
        "all_exactly_21_sessions": True,
    }


def construct_sp500_first(
    sampling: pd.DataFrame,
    loaded_series: dict[str, LoadedTargetSeries],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Construct the required first target before expanding to all twelve."""
    return _target_values("sp500", loaded_series["sp500"], sampling)


def construct_all_market_targets(
    sampling: pd.DataFrame,
    loaded_series: dict[str, LoadedTargetSeries],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the canonical target table and a separate endpoint audit table."""
    output = sampling.loc[:, list(TARGET_KEY_COLUMNS)].copy()
    audit = sampling.loc[
        :,
        [
            "sample_date",
            "state_cutoff_date",
            "target_baseline_date",
            "target_end_date",
            "target_horizon_sessions",
            "event_types_json",
            "event_ids_json",
        ],
    ].copy()

    # The S&P 500 is deliberately constructed first, then reused when the
    # remaining target family is expanded.
    sp500_baseline, sp500_future, sp500_target = construct_sp500_first(
        sampling, loaded_series
    )
    output["sp500"] = sp500_target
    audit["sp500_baseline_value"] = sp500_baseline
    audit["sp500_future_value"] = sp500_future
    audit["sp500_calculated_target"] = sp500_target

    for target_name in TARGET_NAMES:
        if target_name == "sp500":
            continue
        baseline, future, target = _target_values(
            target_name, loaded_series[target_name], sampling
        )
        output[target_name] = target
        audit[f"{target_name}_baseline_value"] = baseline
        audit[f"{target_name}_future_value"] = future
        audit[f"{target_name}_calculated_target"] = target

    output = output.loc[:, list(TARGET_OUTPUT_COLUMNS)]
    if output[list(TARGET_NAMES)].isna().any().any():
        raise ValueError("Constructed target table contains missing labels.")
    if not output[list(TARGET_NAMES)].map(math.isfinite).all().all():
        raise ValueError("Constructed target table contains non-finite labels.")
    return output, audit


def _decimal_yahoo_values(path: Path, value_field: str) -> dict[str, Decimal]:
    values: dict[str, Decimal] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_value = row.get(value_field)
            if raw_value in (None, ""):
                continue
            values[str(row["Date"])] = Decimal(str(raw_value))
    return values


def _decimal_fred_values(path: Path) -> dict[str, Decimal]:
    payload = _read_json(path)
    return {
        str(row["date"]): Decimal(str(row["value"]))
        for row in payload.get("observations", [])
        if row.get("value") not in (None, "", ".")
    }


def _manual_check_positions(target_values: pd.Series) -> list[int]:
    row_count = len(target_values)
    maximum_absolute = int(target_values.abs().idxmax())
    positions = {
        0,
        row_count // 4,
        row_count // 2,
        (3 * row_count) // 4,
        row_count - 1,
        maximum_absolute,
    }
    return sorted(positions)


def build_manual_checks(
    sampling: pd.DataFrame,
    targets: pd.DataFrame,
    loaded_series: dict[str, LoadedTargetSeries],
) -> pd.DataFrame:
    """Independently recompute representative S&P and yield rows with Decimal."""
    records: list[dict[str, Any]] = []

    sp500_loaded = loaded_series["sp500"]
    sp500_raw = _decimal_yahoo_values(
        sp500_loaded.path, sp500_loaded.spec.value_field
    )
    for position in _manual_check_positions(targets["sp500"]):
        row = sampling.iloc[position]
        baseline_date = str(row["target_baseline_date"])
        future_date = str(row["target_end_date"])
        baseline = sp500_raw[baseline_date]
        future = sp500_raw[future_date]
        hand = Decimal(100) * (future / baseline - Decimal(1))
        vectorized = Decimal(str(targets.iloc[position]["sp500"]))
        difference = abs(hand - vectorized)
        records.append(
            {
                "target": "sp500",
                "sample_date": row["sample_date"],
                "current_date": baseline_date,
                "current_value": str(baseline),
                "future_date": future_date,
                "future_value": str(future),
                "hand_calculated_target": str(hand),
                "vectorized_target": str(vectorized),
                "absolute_difference": str(difference),
                "unit": "percent",
                "passed": difference <= Decimal("1e-10"),
            }
        )

    yield_loaded = loaded_series["ten_year_yield_bps"]
    yield_raw = _decimal_fred_values(yield_loaded.path)
    yield_target = targets["ten_year_yield_bps"]
    yield_positions = {
        0,
        int(yield_target.idxmin()),
        int(yield_target.idxmax()),
        len(yield_target) - 1,
    }
    for position in sorted(yield_positions):
        row = sampling.iloc[position]
        baseline_date = str(row["target_baseline_date"])
        future_date = str(row["target_end_date"])
        baseline = yield_raw[baseline_date]
        future = yield_raw[future_date]
        hand = Decimal(100) * (future - baseline)
        vectorized = Decimal(
            str(targets.iloc[position]["ten_year_yield_bps"])
        )
        difference = abs(hand - vectorized)
        records.append(
            {
                "target": "ten_year_yield_bps",
                "sample_date": row["sample_date"],
                "current_date": baseline_date,
                "current_value": str(baseline),
                "future_date": future_date,
                "future_value": str(future),
                "hand_calculated_target": str(hand),
                "vectorized_target": str(vectorized),
                "absolute_difference": str(difference),
                "unit": "basis_points",
                "passed": difference <= Decimal("1e-10"),
            }
        )

    checks = pd.DataFrame.from_records(records)
    if not checks["passed"].all():
        failed = checks.loc[~checks["passed"]].to_dict("records")
        raise ValueError(f"Independent manual target checks failed: {failed}")
    return checks


def _rounded(value: Any, digits: int = 10) -> float:
    return round(float(value), digits)


def build_distribution_summary(
    targets: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for target_name in TARGET_NAMES:
        values = pd.to_numeric(targets[target_name], errors="coerce")
        maximum_absolute_position = int(values.abs().idxmax())
        summary[target_name] = {
            "unit": (
                "basis_points"
                if target_name == "ten_year_yield_bps"
                else "percent"
            ),
            "count": int(values.count()),
            "missing_count": int(values.isna().sum()),
            "non_finite_count": int(
                sum(not math.isfinite(value) for value in values.dropna())
            ),
            "mean": _rounded(values.mean()),
            "standard_deviation": _rounded(values.std(ddof=1)),
            "minimum": _rounded(values.min()),
            "p01": _rounded(values.quantile(0.01)),
            "p05": _rounded(values.quantile(0.05)),
            "median": _rounded(values.median()),
            "p95": _rounded(values.quantile(0.95)),
            "p99": _rounded(values.quantile(0.99)),
            "maximum": _rounded(values.max()),
            "maximum_absolute_value": _rounded(values.abs().max()),
            "maximum_absolute_sample_date": str(
                targets.iloc[maximum_absolute_position]["sample_date"]
            ),
            "review_threshold_absolute": EXTREME_REVIEW_THRESHOLDS[
                target_name
            ],
            "threshold_exceedance_count": int(
                (
                    values.abs()
                    > EXTREME_REVIEW_THRESHOLDS[target_name]
                ).sum()
            ),
        }
    return summary


def build_extreme_value_review(
    targets: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target_name in TARGET_NAMES:
        threshold = EXTREME_REVIEW_THRESHOLDS[target_name]
        positions = targets.index[targets[target_name].abs() > threshold]
        for position in positions:
            records.append(
                {
                    "target": target_name,
                    "sample_date": targets.loc[position, "sample_date"],
                    "target_baseline_date": targets.loc[
                        position, "target_baseline_date"
                    ],
                    "target_end_date": targets.loc[
                        position, "target_end_date"
                    ],
                    "baseline_value": audit.loc[
                        position, f"{target_name}_baseline_value"
                    ],
                    "future_value": audit.loc[
                        position, f"{target_name}_future_value"
                    ],
                    "target_value": targets.loc[position, target_name],
                    "unit": (
                        "basis_points"
                        if target_name == "ten_year_yield_bps"
                        else "percent"
                    ),
                    "absolute_review_threshold": threshold,
                    "action": "retained_for_documented_review",
                }
            )
    columns = (
        "target",
        "sample_date",
        "target_baseline_date",
        "target_end_date",
        "baseline_value",
        "future_value",
        "target_value",
        "unit",
        "absolute_review_threshold",
        "action",
    )
    return pd.DataFrame.from_records(records, columns=columns)


def _source_manifest(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [
        {"path": _relative(path), "sha256": _sha256(path)}
        for path in sorted(set(path.resolve() for path in paths))
    ]


def build_target_artifacts(
    *,
    selected_sampling_path: Path = SELECTED_SAMPLING_PATH,
    raw_root: Path = RAW_ROOT,
) -> dict[str, Any]:
    """Build all Step 5 artifacts in memory using local files only."""
    sampling = load_selected_sampling_index(selected_sampling_path)
    loaded_series, source_paths = load_target_series(raw_root)
    horizon_validation = _validate_exact_horizons(sampling, raw_root)

    # Acceptance order: build S&P first, independently check it, then expand.
    sp500_baseline, sp500_future, sp500_target = construct_sp500_first(
        sampling, loaded_series
    )
    if len(sp500_target) != len(sampling):
        raise ValueError("S&P first-pass target changed the selected row count.")

    targets, audit = construct_all_market_targets(sampling, loaded_series)
    manual_checks = build_manual_checks(sampling, targets, loaded_series)
    distributions = build_distribution_summary(targets)
    extremes = build_extreme_value_review(targets, audit)

    validations = {
        "selected_sampling_design_is_event_aware": bool(
            (sampling["sampling_design"] == "EVENT_AWARE").all()
        ),
        "input_rows_preserved_without_dropping": len(targets) == len(sampling),
        "all_target_baselines_equal_state_cutoffs": bool(
            (
                targets["target_baseline_date"]
                == targets["state_cutoff_date"]
            ).all()
        ),
        "all_target_windows_are_exactly_21_sessions": bool(
            horizon_validation["all_exactly_21_sessions"]
        ),
        "all_12_contract_targets_present": tuple(
            column
            for column in targets.columns
            if column in TARGET_NAMES
        )
        == TARGET_NAMES,
        "all_exact_endpoint_lookups_succeeded": True,
        "no_target_values_missing": bool(
            not targets[list(TARGET_NAMES)].isna().any().any()
        ),
        "all_target_values_finite": bool(
            targets[list(TARGET_NAMES)].map(math.isfinite).all().all()
        ),
        "all_manual_checks_passed": bool(manual_checks["passed"].all()),
        "yield_basis_point_conversion_checked": bool(
            (
                manual_checks["target"] == "ten_year_yield_bps"
            ).sum()
            >= 3
        ),
        "no_forward_fill_backfill_or_endpoint_shift": True,
        "source_field_policies_match_step_3": all(
            (
                spec.value_field == "Adj Close"
                if spec.target
                in {
                    "gold",
                    "technology",
                    "energy",
                    "financials",
                    "utilities",
                    "healthcare",
                    "consumer_discretionary",
                }
                else spec.value_field in {"Close", "value"}
            )
            for spec in TARGET_SPECS
        ),
    }
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise ValueError(f"Step 5 target validation failed: {failed}")

    target_specs = []
    for spec in TARGET_SPECS:
        target_specs.append(
            {
                "target": spec.target,
                "source": spec.source,
                "path": f"data/scenario_playground/raw/{spec.relative_path}",
                "field": spec.value_field,
                "value_policy": spec.value_policy,
                "formula": (
                    "100 * (future_yield_pct - baseline_yield_pct)"
                    if spec.target == "ten_year_yield_bps"
                    else "100 * (future_value / baseline_value - 1)"
                ),
                "unit": (
                    "basis_points"
                    if spec.target == "ten_year_yield_bps"
                    else "percent"
                ),
            }
        )

    summary = {
        "schema_version": "scenario_market_targets_v1",
        "construction_mode": "offline_cached_data_only",
        "sampling_design": "EVENT_AWARE",
        "input_sampling_path": _relative(selected_sampling_path),
        "input_row_count": int(len(sampling)),
        "output_row_count": int(len(targets)),
        "first_sample_date": str(targets["sample_date"].iloc[0]),
        "last_sample_date": str(targets["sample_date"].iloc[-1]),
        "first_target_baseline_date": str(
            targets["target_baseline_date"].iloc[0]
        ),
        "last_target_end_date": str(targets["target_end_date"].iloc[-1]),
        "target_horizon_sessions": TARGET_HORIZON_SESSIONS,
        "target_names": list(TARGET_NAMES),
        "target_output_schema": list(TARGET_OUTPUT_COLUMNS),
        "target_audit_schema": list(audit.columns),
        "manual_check_schema": list(manual_checks.columns),
        "extreme_review_schema": list(extremes.columns),
        "target_specs": target_specs,
        "horizon_validation": horizon_validation,
        "manual_check_count": int(len(manual_checks)),
        "sp500_manual_check_count": int(
            (manual_checks["target"] == "sp500").sum()
        ),
        "yield_manual_check_count": int(
            (manual_checks["target"] == "ten_year_yield_bps").sum()
        ),
        "manual_checks": manual_checks.to_dict("records"),
        "distributions": distributions,
        "extreme_review_count": int(len(extremes)),
        "extreme_review_by_target": dict(
            sorted(Counter(extremes["target"]).items())
        )
        if len(extremes)
        else {},
        "validations": validations,
        "source_manifest": _source_manifest(
            [selected_sampling_path] + source_paths
        ),
        "scope_boundary": (
            "No shocks, state features, splitting, modeling, Supabase, backend, "
            "or frontend work is included."
        ),
    }
    return {
        "targets": targets,
        "audit": audit,
        "manual_checks": manual_checks,
        "extremes": extremes,
        "summary": summary,
    }


def persist_target_artifacts(
    artifacts: dict[str, Any],
    *,
    output_dir: Path = TARGET_OUTPUT_ROOT,
) -> dict[str, str]:
    output_dir = output_dir.resolve()
    paths = {
        "targets": output_dir / "market_targets.csv",
        "audit": output_dir / "market_target_audit.csv",
        "manual_checks": output_dir / "manual_target_checks.csv",
        "extremes": output_dir / "market_target_extremes.csv",
        "summary": output_dir / "market_target_summary.json",
    }
    csv_text = {
        name: _frame_to_csv_text(artifacts[name])
        for name in ("targets", "audit", "manual_checks", "extremes")
    }
    csv_hashes = {
        paths[name].name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in csv_text.items()
    }
    summary = dict(artifacts["summary"])
    summary["output_files"] = {
        name: {
            "path": _relative(path),
            "sha256": csv_hashes.get(path.name),
        }
        for name, path in paths.items()
    }
    summary_text = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    )
    for name, text in csv_text.items():
        write_text_atomic(paths[name], text)
    write_text_atomic(paths["summary"], summary_text)
    return {name: str(path) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Construct all 12 Scenario Playground 21-session market targets "
            "from the selected Step 4 index and local Step 3 caches."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate in memory without writing interim outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TARGET_OUTPUT_ROOT,
        help=(
            "Output directory (default: "
            "data/scenario_playground/interim/targets)."
        ),
    )
    args = parser.parse_args(argv)
    artifacts = build_target_artifacts()
    outputs = (
        {}
        if args.dry_run
        else persist_target_artifacts(
            artifacts, output_dir=args.output_dir
        )
    )
    summary = artifacts["summary"]
    result = {
        "sampling_design": summary["sampling_design"],
        "row_count": summary["output_row_count"],
        "target_count": len(summary["target_names"]),
        "manual_checks": summary["manual_check_count"],
        "extreme_review_count": summary["extreme_review_count"],
        "outputs": outputs,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
