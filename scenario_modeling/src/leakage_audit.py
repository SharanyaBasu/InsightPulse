"""Step 9 exploratory data-quality and leakage audit.

The audit uses only completed local artifacts.  Final-test target values are
excluded from distributions, correlations, outlier review, and all decisions.
The final-test period receives structural checks only (schema, dates,
missingness, and lock status).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .cache import write_text_atomic
from .config import INTERIM_ROOT, RAW_ROOT
from .feature_sources import frame_to_csv_text, relative_path, source_manifest
from .model_matrix import (
    FINAL_AUDIT_EXCLUDED_FEATURES,
    FINAL_FEATURE_NAMES,
    INITIAL_FEATURE_NAMES,
    MATRIX_OUTPUT_ROOT,
    OIL_VOLATILITY_FEATURE,
    STATE_PATH,
    TARGET_NAMES,
    build_model_matrix_artifacts,
    persist_model_matrix_artifacts,
)
from .shock_features import (
    CONTRACT_TO_HISTORICAL_FEATURE,
)


STATE_AVAILABILITY_AUDIT_PATH = (
    INTERIM_ROOT
    / "features"
    / "state"
    / "state_feature_availability_audit.csv"
)
SHOCK_EVENT_AUDIT_PATH = (
    INTERIM_ROOT
    / "features"
    / "shocks"
    / "scenario_shock_event_audit.csv"
)
SHOCK_MARKET_AUDIT_PATH = (
    INTERIM_ROOT
    / "features"
    / "shocks"
    / "scenario_market_shock_audit.csv"
)
RAW_OIL_PATH = RAW_ROOT / "yahoo" / "CL_F.csv"
AUDIT_OUTPUT_ROOT = INTERIM_ROOT / "audits" / "leakage"

SHOCK_RANGE_COLUMNS = (
    "contract_input",
    "historical_feature",
    "unit",
    "historical_representation",
    "proxy",
    "scope",
    "row_count",
    "active_event_or_window_count",
    "zero_count",
    "nonzero_count",
    "minimum",
    "p01",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "maximum",
    "active_minimum",
    "active_median",
    "active_maximum",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required audit source is missing: {path}")
    return pd.read_csv(path)


def _finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[numeric.map(lambda value: pd.notna(value) and math.isfinite(value))]


def _distribution_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    role: str,
    scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        finite = _finite_numeric(numeric)
        quantiles = finite.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        q1 = float(quantiles.loc[0.25])
        q3 = float(quantiles.loc[0.75])
        iqr = q3 - q1
        unique_count = int(finite.nunique())
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((finite < lower) | (finite > upper)).sum())
            outlier_rule = "1.5_IQR_REVIEW_FLAG"
        else:
            lower = q1
            upper = q3
            outlier_count = 0
            outlier_rule = "NO_IQR_FLAG_ZERO_IQR"
        rows.append(
            {
                "column_name": column,
                "column_role": role,
                "scope": scope,
                "count": int(len(numeric)),
                "finite_count": int(len(finite)),
                "missing_count": int(numeric.isna().sum()),
                "non_finite_count": int(
                    numeric.notna().sum() - len(finite)
                ),
                "zero_count": int((finite.abs() <= 1e-12).sum()),
                "unique_count": unique_count,
                "minimum": float(finite.min()),
                "p01": float(quantiles.loc[0.01]),
                "p05": float(quantiles.loc[0.05]),
                "p25": q1,
                "median": float(quantiles.loc[0.50]),
                "p75": q3,
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "maximum": float(finite.max()),
                "mean": float(finite.mean()),
                "standard_deviation": float(finite.std(ddof=1)),
                "iqr_lower_review_bound": float(lower),
                "iqr_upper_review_bound": float(upper),
                "iqr_outlier_count": outlier_count,
                "outlier_rule": outlier_rule,
            }
        )
    return pd.DataFrame(rows)


def _within_correlations(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    left_role: str,
    scope: str,
) -> pd.DataFrame:
    correlation = frame.loc[:, list(columns)].corr(method="pearson")
    rows: list[dict[str, Any]] = []
    for left_position, left in enumerate(columns):
        for right in columns[left_position + 1 :]:
            value = float(correlation.loc[left, right])
            rows.append(
                {
                    "left_column": left,
                    "right_column": right,
                    "left_role": left_role,
                    "right_role": left_role,
                    "scope": scope,
                    "pearson_correlation": value,
                    "absolute_correlation": abs(value),
                    "high_absolute_correlation_flag": abs(value) >= 0.80,
                }
            )
    return pd.DataFrame(rows)


def _cross_correlations(
    frame: pd.DataFrame,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    *,
    scope: str,
) -> pd.DataFrame:
    correlation = frame.loc[
        :, [*left_columns, *right_columns]
    ].corr(method="pearson")
    rows = []
    for left in left_columns:
        for right in right_columns:
            value = float(correlation.loc[left, right])
            rows.append(
                {
                    "feature": left,
                    "target": right,
                    "scope": scope,
                    "pearson_correlation": value,
                    "absolute_correlation": abs(value),
                    "high_absolute_correlation_flag": abs(value) >= 0.50,
                }
            )
    return pd.DataFrame(rows)


def _missingness_by_year(
    matrix: pd.DataFrame,
    split_manifest: pd.DataFrame,
    value_columns: Sequence[str],
    feature_names: set[str],
) -> pd.DataFrame:
    joined = matrix.merge(
        split_manifest.loc[:, ["sample_date", "raw_period"]],
        on="sample_date",
        how="inner",
        validate="one_to_one",
    )
    joined["calendar_year"] = pd.to_datetime(
        joined["sample_date"]
    ).dt.year
    rows = []
    for (year, period), group in joined.groupby(
        ["calendar_year", "raw_period"], sort=True
    ):
        for column in value_columns:
            missing = int(group[column].isna().sum())
            rows.append(
                {
                    "calendar_year": int(year),
                    "raw_period": str(period),
                    "column_name": column,
                    "column_role": (
                        "feature" if column in feature_names else "target"
                    ),
                    "row_count": int(len(group)),
                    "missing_count": missing,
                    "missing_rate": float(missing / len(group)),
                    "check_type": (
                        "STRUCTURAL_ONLY_FINAL_TEST"
                        if period == "final_test"
                        else "DEVELOPMENT_DATA_QUALITY"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _regime_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(frame["sample_date"])
    calendar_regimes = pd.Series(index=frame.index, dtype="object")
    calendar_regimes.loc[dates < "2020-01-01"] = "pre_pandemic"
    calendar_regimes.loc[
        (dates >= "2020-01-01") & (dates <= "2020-12-31")
    ] = "pandemic_shock"
    calendar_regimes.loc[
        (dates >= "2021-01-01") & (dates <= "2022-12-31")
    ] = "recovery_and_inflation"
    calendar_regimes.loc[dates >= "2023-01-01"] = "later_pretest"

    vix = frame["vix_level_points"]
    vix_regimes = pd.cut(
        vix,
        bins=[-math.inf, 15.0, 25.0, math.inf],
        labels=["vix_below_15", "vix_15_to_25", "vix_above_25"],
        right=False,
    )
    rate = frame["fed_funds_target_midpoint_pct"]
    rate_regimes = pd.cut(
        rate,
        bins=[-math.inf, 0.5, 3.0, math.inf],
        labels=[
            "policy_below_0_5",
            "policy_0_5_to_3",
            "policy_at_least_3",
        ],
        right=False,
    )

    rows: list[dict[str, Any]] = []
    for family, values in {
        "calendar_market_regime": calendar_regimes,
        "vix_regime": vix_regimes.astype(str),
        "policy_rate_regime": rate_regimes.astype(str),
    }.items():
        for regime, indexes in values.groupby(values, observed=True).groups.items():
            selected_dates = dates.loc[list(indexes)]
            rows.append(
                {
                    "regime_family": family,
                    "regime": str(regime),
                    "row_count": int(len(indexes)),
                    "first_sample_date": selected_dates.min().date().isoformat(),
                    "last_sample_date": selected_dates.max().date().isoformat(),
                    "scope": "PRE_FINAL_TEST_FEATURES_ONLY",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["regime_family", "regime"]
    ).reset_index(drop=True)


def _shock_range_report(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    representations = {
        "fed_funds_rate_change_bps": (
            "Official target midpoint change on attached FOMC rows",
            "basis_points",
            "fomc_release_flag",
            False,
        ),
        "cpi_surprise_pct": (
            "Initial headline CPI MoM minus latest strictly pre-release Cleveland nowcast",
            "percentage_points",
            "cpi_release_flag",
            False,
        ),
        "oil_price_change_pct": (
            "WTI spot change over the stored state-cutoff-to-sample window",
            "percent",
            None,
            False,
        ),
        "gdp_growth_surprise_pct": (
            "BEA advance real GDP QoQ SAAR minus GDPNow right-before forecast",
            "annualized_percentage_points",
            "gdp_release_flag",
            False,
        ),
        "unemployment_change_pct": (
            "Initial unemployment rate minus prior initial release",
            "percentage_points",
            "unemployment_release_flag",
            False,
        ),
        "pmi_change_points": (
            "Change in the explicit Kansas City Fed PMI-style regional proxy",
            "pmi_style_index_points",
            "pmi_release_flag",
            True,
        ),
        "dxy_change_pct": (
            "DXY Close change over the stored state-cutoff-to-sample window",
            "percent",
            None,
            False,
        ),
        "vix_change_points": (
            "VIXCLS point change over the stored state-cutoff-to-sample window",
            "vix_index_points",
            None,
            False,
        ),
    }
    rows = []
    for contract_name, (
        representation,
        unit,
        flag,
        proxy,
    ) in representations.items():
        historical_name = CONTRACT_TO_HISTORICAL_FEATURE[contract_name]
        values = _finite_numeric(frame[historical_name])
        if flag is None:
            active = values
            active_count = len(values)
        else:
            active_mask = frame[flag].astype(int) == 1
            active = _finite_numeric(frame.loc[active_mask, historical_name])
            active_count = int(active_mask.sum())
        quantiles = values.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        rows.append(
            {
                "contract_input": contract_name,
                "historical_feature": historical_name,
                "unit": unit,
                "historical_representation": representation,
                "proxy": proxy,
                "scope": "PRE_FINAL_TEST_FEATURES_ONLY",
                "row_count": int(len(values)),
                "active_event_or_window_count": int(active_count),
                "zero_count": int((values.abs() <= 1e-12).sum()),
                "nonzero_count": int((values.abs() > 1e-12).sum()),
                "minimum": float(values.min()),
                "p01": float(quantiles.loc[0.01]),
                "p05": float(quantiles.loc[0.05]),
                "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.50]),
                "p75": float(quantiles.loc[0.75]),
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "maximum": float(values.max()),
                "active_minimum": float(active.min()),
                "active_median": float(active.median()),
                "active_maximum": float(active.max()),
            }
        )
    return pd.DataFrame(rows, columns=SHOCK_RANGE_COLUMNS)


def _extreme_observations(
    feature_scope: pd.DataFrame,
    target_scope: pd.DataFrame,
    feature_names: Sequence[str],
    target_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, frame, columns in (
        ("feature", feature_scope, feature_names),
        ("target", target_scope, target_names),
    ):
        for column in columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            for kind, position in (
                ("minimum", numeric.idxmin()),
                ("maximum", numeric.idxmax()),
            ):
                row = frame.loc[position]
                rows.append(
                    {
                        "column_role": role,
                        "column_name": column,
                        "extreme_type": kind,
                        "sample_date": row["sample_date"],
                        "value": float(row[column]),
                        "event_types_json": row["event_types_json"],
                        "scope": (
                            "PRE_FINAL_TEST_FEATURES_ONLY"
                            if role == "feature"
                            else "PURGED_PRE_FINAL_TEST_TARGET_AUDIT"
                        ),
                        "action": "RETAINED_RAW_FOR_REVIEW",
                    }
                )
    return pd.DataFrame(rows)


def _oil_volatility_review(
    initial_matrix: pd.DataFrame,
    final_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = _read_csv(STATE_PATH)
    availability = _read_csv(STATE_AVAILABILITY_AUDIT_PATH)
    raw = _read_csv(RAW_OIL_PATH)
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.date
    raw["Close"] = pd.to_numeric(raw["Close"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Close"]).sort_values("Date")
    raw = raw.drop_duplicates("Date", keep="last").reset_index(drop=True)

    nonpositive = raw[raw["Close"] <= 0]
    if len(nonpositive) != 1:
        raise ValueError(
            "Expected exactly one cached nonpositive CL=F close for the "
            f"documented audit, found {len(nonpositive)}."
        )
    nonpositive_row = nonpositive.iloc[0]
    nonpositive_date = nonpositive_row["Date"]
    nonpositive_close = float(nonpositive_row["Close"])
    dates = raw["Date"].tolist()
    positions = {value: position for position, value in enumerate(dates)}
    nonpositive_position = positions[nonpositive_date]

    merged = state.loc[
        :, [
            "sample_date",
            "state_cutoff_date",
            OIL_VOLATILITY_FEATURE,
        ]
    ].merge(
        availability.loc[:, ["sample_date", "oil_source_date"]],
        on="sample_date",
        how="inner",
        validate="one_to_one",
    )
    retained_samples = set(initial_matrix["sample_date"])
    merged = merged[merged["sample_date"].isin(retained_samples)].copy()
    records = []
    for row in merged.itertuples(index=False):
        source_date = pd.Timestamp(row.oil_source_date).date()
        source_position = positions[source_date]
        contains_nonpositive = (
            source_position - 20
            <= nonpositive_position
            <= source_position
        )
        if contains_nonpositive:
            records.append(
                {
                    "sample_date": row.sample_date,
                    "state_cutoff_date": row.state_cutoff_date,
                    "oil_source_date": row.oil_source_date,
                    "feature_name": OIL_VOLATILITY_FEATURE,
                    "feature_value": float(
                        getattr(row, OIL_VOLATILITY_FEATURE)
                    ),
                    "rolling_window_start_source_date": dates[
                        source_position - 20
                    ].isoformat(),
                    "rolling_window_end_source_date": source_date.isoformat(),
                    "nonpositive_source_date": nonpositive_date.isoformat(),
                    "nonpositive_close": nonpositive_close,
                    "rolling_window_contains_nonpositive_close": True,
                    "decision": "EXCLUDE_FEATURE_RETAIN_ROW",
                    "reason": (
                        "Arithmetic percentage returns are not stable when the "
                        "denominator and settlement cross zero."
                    ),
                }
            )
    review = pd.DataFrame(records)
    if review.empty:
        raise ValueError("No selected state windows captured the CL=F issue.")
    if OIL_VOLATILITY_FEATURE not in initial_matrix:
        raise ValueError("The pre-audit matrix is missing the oil feature.")
    if OIL_VOLATILITY_FEATURE in final_matrix:
        raise ValueError("The corrected matrix retained the rejected oil feature.")

    summary = {
        "raw_nonpositive_close_count": int(len(nonpositive)),
        "nonpositive_source_date": nonpositive_date.isoformat(),
        "nonpositive_close": nonpositive_close,
        "affected_selected_row_count": int(len(review)),
        "first_affected_sample": str(review["sample_date"].iloc[0]),
        "last_affected_sample": str(review["sample_date"].iloc[-1]),
        "maximum_affected_feature_value": float(
            review["feature_value"].max()
        ),
        "decision": "EXCLUDE_FEATURE_RETAIN_ALL_ROWS",
        "decision_reason": (
            "The data event is real, but percentage-return volatility through "
            "a negative futures settlement is not a coherent comparable scale. "
            "Global clipping would learn from all periods and hide the source "
            "problem; deleting rows would discard valid pandemic events."
        ),
        "no_clipping_winsorization_or_row_removal": True,
    }
    return review, summary


def _checklist_row(
    name: str,
    passed: bool,
    evidence: str,
    category: str,
) -> dict[str, Any]:
    return {
        "check_name": name,
        "category": category,
        "passed": bool(passed),
        "evidence": evidence,
    }


def build_leakage_audit_artifacts(
    *,
    initial_matrix_artifacts: dict[str, Any] | None = None,
    final_matrix_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the pre-correction matrix and re-audit the corrected matrix."""
    initial = initial_matrix_artifacts or build_model_matrix_artifacts(
        excluded_features=()
    )
    final = final_matrix_artifacts or build_model_matrix_artifacts(
        excluded_features=FINAL_AUDIT_EXCLUDED_FEATURES
    )
    initial_matrix = initial["matrix"]
    matrix = final["matrix"]
    provenance = final["provenance"]
    split_manifest = final["split_manifest"]
    feature_names = tuple(final["summary"]["feature_names"])
    target_names = tuple(final["summary"]["target_names"])

    if tuple(feature_names) != FINAL_FEATURE_NAMES:
        raise ValueError("The corrected matrix feature schema is not final.")
    expected_initial_columns = (
        ["sample_date", *INITIAL_FEATURE_NAMES, *TARGET_NAMES]
    )
    if initial_matrix.columns.tolist() != expected_initial_columns:
        raise ValueError("The pre-audit matrix does not contain 59 features.")
    expected_final_from_initial = initial_matrix.drop(
        columns=[OIL_VOLATILITY_FEATURE]
    )
    correction_only_removed_feature = expected_final_from_initial.equals(
        matrix
    )
    if not correction_only_removed_feature:
        raise ValueError(
            "The audit correction changed values or rows beyond the approved "
            "oil-volatility feature exclusion."
        )

    joined = matrix.merge(
        provenance.loc[
            :, ["sample_date", "event_types_json", "event_ids_json"]
        ],
        on="sample_date",
        how="inner",
        validate="one_to_one",
    ).merge(
        split_manifest.loc[
            :,
            [
                "sample_date",
                "raw_period",
                "modeling_role",
                "eligible_for_period_evaluation",
                "target_access_policy",
            ],
        ],
        on="sample_date",
        how="inner",
        validate="one_to_one",
    )

    # Final-test targets are intentionally excluded from every value-based
    # table below. Structural missingness checks remain allowed.
    feature_scope = joined[joined["raw_period"] != "final_test"].copy()
    target_scope = joined[
        joined["modeling_role"].isin(["development_train", "validation"])
    ].copy()
    feature_scope_label = "PRE_FINAL_TEST_FEATURES_ONLY"
    target_scope_label = "PURGED_PRE_FINAL_TEST_TARGET_AUDIT"

    feature_distributions = _distribution_table(
        feature_scope,
        feature_names,
        role="feature",
        scope=feature_scope_label,
    )
    target_distributions = _distribution_table(
        target_scope,
        target_names,
        role="target",
        scope=target_scope_label,
    )
    feature_correlations = _within_correlations(
        feature_scope,
        feature_names,
        left_role="feature",
        scope=feature_scope_label,
    )
    target_correlations = _within_correlations(
        target_scope,
        target_names,
        left_role="target",
        scope=target_scope_label,
    )
    feature_target_correlations = _cross_correlations(
        target_scope,
        feature_names,
        target_names,
        scope=target_scope_label,
    )
    missingness_by_year = _missingness_by_year(
        matrix,
        split_manifest,
        [*feature_names, *target_names],
        set(feature_names),
    )
    regime_coverage = _regime_coverage(feature_scope)
    shock_ranges = _shock_range_report(feature_scope)
    extreme_observations = _extreme_observations(
        feature_scope,
        target_scope,
        feature_names,
        target_names,
    )
    oil_review, oil_summary = _oil_volatility_review(
        initial_matrix, matrix
    )

    state_availability = _read_csv(STATE_AVAILABILITY_AUDIT_PATH)
    state_date_columns = [
        column
        for column in state_availability.columns
        if column.endswith("_availability_date")
        or column.endswith("_source_date")
    ]
    state_dates_before_cutoff = all(
        (
            pd.to_datetime(state_availability[column])
            <= pd.to_datetime(state_availability["state_cutoff_date"])
        ).all()
        for column in state_date_columns
    )

    shock_events = _read_csv(SHOCK_EVENT_AUDIT_PATH)
    event_expectations = shock_events[
        shock_events["expectation_date"].notna()
    ]
    event_timing_pass = bool(
        shock_events["point_in_time_pass"].astype(bool).all()
        and (
            pd.to_datetime(shock_events["model_availability_date"])
            <= pd.to_datetime(shock_events["sample_date"])
        ).all()
        and (
            pd.to_datetime(shock_events["state_cutoff_date"])
            < pd.to_datetime(shock_events["model_availability_date"])
        ).all()
        and (
            pd.to_datetime(event_expectations["expectation_date"])
            < pd.to_datetime(event_expectations["original_release_date"])
        ).all()
    )
    market_shocks = _read_csv(SHOCK_MARKET_AUDIT_PATH)
    market_timing_pass = bool(
        market_shocks["point_in_time_pass"].astype(bool).all()
        and (
            pd.to_datetime(market_shocks["window_start_source_date"])
            <= pd.to_datetime(market_shocks["window_start_boundary"])
        ).all()
        and (
            pd.to_datetime(market_shocks["window_end_source_date"])
            <= pd.to_datetime(market_shocks["window_end_boundary"])
        ).all()
    )

    feature_catalog = final["column_catalog"]
    selected_feature_catalog = feature_catalog[
        (feature_catalog["storage_artifact"]
         == "scenario_model_matrix_raw.csv")
        & (feature_catalog["column_role"] == "feature")
    ]
    no_target_feature_sources = bool(
        not selected_feature_catalog["source_artifact"]
        .astype(str)
        .str.contains("/targets/")
        .any()
    )
    final_test_rows = split_manifest[
        split_manifest["raw_period"] == "final_test"
    ]
    final_test_structural_missing = int(
        matrix[
            matrix["sample_date"].isin(final_test_rows["sample_date"])
        ][[*feature_names, *target_names]]
        .isna()
        .sum()
        .sum()
    )
    shock_variation_pass = bool(
        (shock_ranges["nonzero_count"] >= 20).all()
        and (
            shock_ranges["active_event_or_window_count"]
            >= 20
        ).all()
    )
    split_validations = final["summary"]["split"]["validations"]

    checklist_rows = [
        _checklist_row(
            "no_target_derived_features",
            no_target_feature_sources,
            (
                "All 58 feature catalog entries originate from Step 6 shocks "
                "or Step 7 state artifacts; target and target-audit files are "
                "not feature sources."
            ),
            "leakage",
        ),
        _checklist_row(
            "no_future_filled_state_values",
            state_dates_before_cutoff,
            (
                f"All {len(state_date_columns)} retained state availability/"
                "source-date columns are on or before state_cutoff_date."
            ),
            "point_in_time",
        ),
        _checklist_row(
            "no_future_event_or_expectation_values",
            event_timing_pass,
            (
                "All 615 structured event records pass their timing check; "
                "every cached expectation date is strictly pre-release."
            ),
            "point_in_time",
        ),
        _checklist_row(
            "no_future_market_shock_endpoints",
            market_timing_pass,
            (
                "All 1,755 oil, DXY, and VIX source endpoints are at or before "
                "their stored shock-window boundary."
            ),
            "point_in_time",
        ),
        _checklist_row(
            "no_scaler_or_learned_preprocessing_fit_on_full_data",
            final["summary"]["validations"][
                "no_imputation_scaling_clipping_or_winsorization"
            ],
            (
                "The final CSV is value-for-value equal to the Step 6-7 source "
                "features after the documented row/column exclusions."
            ),
            "preprocessing",
        ),
        _checklist_row(
            "feature_target_provenance_and_audit_are_separate",
            final["summary"]["validations"][
                "matrix_contains_only_key_features_and_targets"
            ]
            and final["summary"]["validations"][
                "provenance_columns_are_not_model_features"
            ]
            and final["summary"]["validations"][
                "audit_columns_are_not_model_features"
            ],
            (
                "Raw matrix: sample key + 58 features + 12 targets. Timing "
                "provenance, source-row audit, endpoint audit, and split "
                "metadata are separate files."
            ),
            "schema",
        ),
        _checklist_row(
            "genuine_missing_cpi_surprise_resolved_without_imputation",
            bool(
                len(final["row_exclusions"]) == 1
                and final["row_exclusions"]["sample_date"].iloc[0]
                == "2013-08-16"
                and not final["row_exclusions"][
                    "imputation_performed"
                ].astype(bool).any()
            ),
            (
                "The single July-2013 CPI release row is excluded; it is not "
                "zero-filled, future-filled, or imputed."
            ),
            "missingness",
        ),
        _checklist_row(
            "final_matrix_has_no_missing_or_nonfinite_values",
            bool(
                feature_distributions["missing_count"].sum() == 0
                and target_distributions["missing_count"].sum() == 0
                and feature_distributions["non_finite_count"].sum() == 0
                and target_distributions["non_finite_count"].sum() == 0
                and final_test_structural_missing == 0
            ),
            (
                "All 584 rows are structurally complete after the one row "
                "exclusion; no imputation was needed."
            ),
            "missingness",
        ),
        _checklist_row(
            "oil_volatility_structural_issue_corrected",
            bool(
                OIL_VOLATILITY_FEATURE
                not in matrix.columns
                and OIL_VOLATILITY_FEATURE in initial_matrix.columns
                and oil_summary[
                    "no_clipping_winsorization_or_row_removal"
                ]
            ),
            (
                f"The one negative CL=F close affects "
                f"{oil_summary['affected_selected_row_count']} selected rolling "
                "windows. The incoherent volatility feature is excluded while "
                "all rows and other oil features remain raw."
            ),
            "outlier_quality",
        ),
        _checklist_row(
            "audit_correction_changed_only_one_feature",
            correction_only_removed_feature,
            (
                "The corrected 58-feature matrix equals the initial 59-feature "
                "matrix after dropping only oil_volatility_20d_annualized_pct."
            ),
            "reproducibility",
        ),
        _checklist_row(
            "all_targets_use_exact_21_canonical_sessions",
            final["summary"]["validations"][
                "all_targets_span_exactly_21_canonical_sessions"
            ],
            (
                "Every target_end_date is exactly 21 observed ^GSPC sessions "
                "after target_baseline_date."
            ),
            "target_integrity",
        ),
        _checklist_row(
            "development_labels_do_not_cross_validation",
            split_validations[
                "development_labels_end_before_validation_baseline"
            ],
            (
                "Four development rows are purged by their target_end_date; "
                "the last retained development baseline is 24 canonical "
                "sessions before the first validation baseline."
            ),
            "split_leakage",
        ),
        _checklist_row(
            "validation_labels_do_not_cross_final_test",
            split_validations[
                "validation_labels_end_before_final_test_baseline"
            ],
            (
                "Two validation rows are purged by their target_end_date; the "
                "last retained validation baseline is 24 canonical sessions "
                "before the first final-test baseline."
            ),
            "split_leakage",
        ),
        _checklist_row(
            "purge_uses_canonical_target_end_dates_not_21_event_rows",
            split_validations[
                "purge_uses_target_end_dates_not_event_row_counts"
            ],
            (
                "The manifest compares each stored target_end_date with the "
                "next period's first canonical target baseline. It purges 4 "
                "and 2 irregular event rows, not 21 rows."
            ),
            "split_leakage",
        ),
        _checklist_row(
            "final_test_is_locked_and_excluded_from_value_based_audit",
            bool(
                len(final_test_rows) == 114
                and (
                    final_test_rows["target_access_policy"]
                    == "LOCKED_UNTIL_FINAL_EVALUATION"
                ).all()
                and feature_scope["raw_period"].ne("final_test").all()
                and target_scope["raw_period"].ne("final_test").all()
            ),
            (
                "All 114 final-test rows are marked locked. Their target values "
                "are absent from distributions, correlations, outlier review, "
                "shock ranges, and decisions; only structural completeness is "
                "checked."
            ),
            "holdout_integrity",
        ),
        _checklist_row(
            "shock_features_have_training_variation",
            shock_variation_pass,
            (
                "Each of the eight user-facing historical representations has "
                "at least 20 active windows/events and at least 20 nonzero "
                "pre-final-test observations."
            ),
            "model_readiness",
        ),
    ]
    checklist = pd.DataFrame(checklist_rows)
    if not checklist["passed"].all():
        failed = checklist.loc[~checklist["passed"], "check_name"].tolist()
        raise ValueError(f"Step 9 leakage audit failed: {failed}")

    high_feature_correlations = feature_correlations[
        feature_correlations["high_absolute_correlation_flag"]
    ]
    high_target_correlations = target_correlations[
        target_correlations["high_absolute_correlation_flag"]
    ]
    high_feature_target_correlations = feature_target_correlations[
        feature_target_correlations["high_absolute_correlation_flag"]
    ]
    summary = {
        "schema_version": "scenario_leakage_audit_v1",
        "construction_mode": "offline_completed_artifacts_only",
        "audit_sequence": {
            "initial_matrix_rows": int(len(initial_matrix)),
            "initial_feature_count": len(INITIAL_FEATURE_NAMES),
            "initial_issue": (
                "oil_volatility_20d_annualized_pct crosses the negative "
                "April 2020 CL=F settlement."
            ),
            "correction": (
                "Exclude that feature only; retain every row and leave source "
                "state data unchanged."
            ),
            "final_matrix_rows": int(len(matrix)),
            "final_feature_count": len(feature_names),
            "re_audit_passed": True,
        },
        "audit_scope": {
            "pre_final_test_feature_rows": int(len(feature_scope)),
            "purged_pre_final_test_target_rows": int(len(target_scope)),
            "final_test_structural_rows": int(len(final_test_rows)),
            "final_test_value_based_analysis_performed": False,
        },
        "feature_distribution_rows": int(len(feature_distributions)),
        "target_distribution_rows": int(len(target_distributions)),
        "feature_pair_correlation_rows": int(len(feature_correlations)),
        "target_pair_correlation_rows": int(len(target_correlations)),
        "feature_target_correlation_rows": int(
            len(feature_target_correlations)
        ),
        "high_absolute_feature_pair_count_at_0_80": int(
            len(high_feature_correlations)
        ),
        "high_absolute_target_pair_count_at_0_80": int(
            len(high_target_correlations)
        ),
        "high_absolute_feature_target_count_at_0_50": int(
            len(high_feature_target_correlations)
        ),
        "missingness_by_year_rows": int(len(missingness_by_year)),
        "regime_coverage_rows": int(len(regime_coverage)),
        "extreme_observation_rows": int(len(extreme_observations)),
        "scenario_shock_range_rows": int(len(shock_ranges)),
        "oil_volatility_review": oil_summary,
        "split": final["summary"]["split"],
        "leakage_check_count": int(len(checklist)),
        "all_leakage_checks_passed": bool(checklist["passed"].all()),
        "training_approval": "APPROVED_FOR_STEP_10_RIDGE_TRAINING",
        "approval_scope": (
            "Use only development_train rows for fitting and validation rows "
            "for initial out-of-sample evaluation. Keep final_test_locked "
            "unread for model selection. Fit every learned preprocessing "
            "operation on development training data only."
        ),
        "remaining_limitations": [
            (
                "The Kansas City Fed manufacturing feature remains an explicit "
                "regional PMI-style proxy rather than national ISM PMI."
            ),
            (
                "CL=F returns and the oil target retain continuous-futures roll "
                "and contract-structure limitations even after excluding the "
                "zero-crossing volatility feature."
            ),
            (
                "Event-aware rows still overlap within each chronological "
                "period. Walk-forward folds must repeat the target-end-date "
                "purge; random cross-validation remains invalid."
            ),
            (
                "Only 354 development-fit and 110 validation-evaluation rows "
                "remain after boundary purges, so XGBoost must stay shallow and "
                "strongly regularized."
            ),
            (
                "High feature and target correlations are expected in market "
                "data and support Ridge regularization; they are not leakage "
                "by themselves."
            ),
        ],
        "validations": {
            str(row["check_name"]): bool(row["passed"])
            for row in checklist.to_dict("records")
        },
        "source_manifest": source_manifest(
            [
                STATE_AVAILABILITY_AUDIT_PATH,
                SHOCK_EVENT_AUDIT_PATH,
                SHOCK_MARKET_AUDIT_PATH,
                RAW_OIL_PATH,
            ]
        ),
        "scope_boundary": (
            "Step 9 only: no final preprocessing, Ridge, XGBoost, predictions, "
            "Supabase, backend, frontend, or server work."
        ),
    }
    return {
        "feature_distributions": feature_distributions,
        "target_distributions": target_distributions,
        "missingness_by_year": missingness_by_year,
        "regime_coverage": regime_coverage,
        "feature_correlations": feature_correlations,
        "target_correlations": target_correlations,
        "feature_target_correlations": feature_target_correlations,
        "scenario_shock_ranges": shock_ranges,
        "extreme_observations": extreme_observations,
        "oil_volatility_review": oil_review,
        "leakage_checklist": checklist,
        "summary": summary,
        "final_matrix_artifacts": final,
    }


def persist_leakage_audit_artifacts(
    artifacts: dict[str, Any],
    *,
    output_dir: Path = AUDIT_OUTPUT_ROOT,
    matrix_output_dir: Path = MATRIX_OUTPUT_ROOT,
) -> dict[str, str]:
    """Persist the corrected matrix first, then its deterministic audit."""
    matrix_paths = persist_model_matrix_artifacts(
        artifacts["final_matrix_artifacts"],
        output_dir=matrix_output_dir,
    )
    paths = {
        "feature_distributions": output_dir / "feature_distributions.csv",
        "target_distributions": output_dir / "target_distributions.csv",
        "missingness_by_year": output_dir / "missingness_by_year.csv",
        "regime_coverage": output_dir / "regime_coverage.csv",
        "feature_correlations": output_dir / "feature_correlations.csv",
        "target_correlations": output_dir / "target_correlations.csv",
        "feature_target_correlations": (
            output_dir / "feature_target_correlations.csv"
        ),
        "scenario_shock_ranges": output_dir / "scenario_shock_ranges.csv",
        "extreme_observations": output_dir / "extreme_observations.csv",
        "oil_volatility_review": output_dir / "oil_volatility_review.csv",
        "leakage_checklist": output_dir / "leakage_checklist.csv",
        "summary": output_dir / "leakage_audit_summary.json",
    }
    dataframe_names = tuple(
        name for name in paths if name != "summary"
    )
    csv_text = {
        name: frame_to_csv_text(artifacts[name])
        for name in dataframe_names
    }
    summary = dict(artifacts["summary"])
    summary["matrix_output_files"] = {
        name: relative_path(Path(path))
        for name, path in matrix_paths.items()
    }
    summary["output_files"] = {
        name: {
            "path": relative_path(path),
            "sha256": (
                hashlib.sha256(csv_text[name].encode("utf-8")).hexdigest()
                if name in csv_text
                else None
            ),
        }
        for name, path in paths.items()
    }
    for name in dataframe_names:
        write_text_atomic(paths[name], csv_text[name])
    write_text_atomic(
        paths["summary"],
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
    )
    combined = {
        f"matrix_{name}": path for name, path in matrix_paths.items()
    }
    combined.update(
        {name: str(path.resolve()) for name, path in paths.items()}
    )
    return combined


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit, correct, regenerate, and re-audit the raw point-in-time "
            "Scenario Playground model matrix using local artifacts only."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the complete audit in memory without writing outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDIT_OUTPUT_ROOT,
        help=(
            "Audit output directory (default: "
            "data/scenario_playground/interim/audits/leakage)."
        ),
    )
    parser.add_argument(
        "--matrix-output-dir",
        type=Path,
        default=MATRIX_OUTPUT_ROOT,
        help=(
            "Corrected matrix output directory (default: "
            "data/scenario_playground/processed/model_matrix)."
        ),
    )
    args = parser.parse_args(argv)
    artifacts = build_leakage_audit_artifacts()
    outputs = (
        {}
        if args.dry_run
        else persist_leakage_audit_artifacts(
            artifacts,
            output_dir=args.output_dir,
            matrix_output_dir=args.matrix_output_dir,
        )
    )
    summary = artifacts["summary"]
    print(
        json.dumps(
            {
                "initial_feature_count": summary["audit_sequence"][
                    "initial_feature_count"
                ],
                "final_feature_count": summary["audit_sequence"][
                    "final_feature_count"
                ],
                "final_row_count": summary["audit_sequence"][
                    "final_matrix_rows"
                ],
                "leakage_check_count": summary["leakage_check_count"],
                "training_approval": summary["training_approval"],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
