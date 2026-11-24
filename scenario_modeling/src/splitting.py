"""Chronological split metadata for the Scenario Playground model matrix.

This module creates split and purge decisions only.  It does not expose target
values to preprocessing, fit a transformer, or train a model.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import pandas as pd


DEVELOPMENT_START = date(2013, 8, 1)
DEVELOPMENT_END = date(2021, 5, 5)
VALIDATION_START = date(2021, 5, 6)
VALIDATION_END = date(2023, 12, 6)
FINAL_TEST_START = date(2023, 12, 7)
FINAL_TEST_END = date(2026, 7, 16)

SPLIT_MANIFEST_COLUMNS = (
    "sample_date",
    "target_baseline_date",
    "target_end_date",
    "raw_period",
    "modeling_role",
    "eligible_for_period_evaluation",
    "purged_for_target_overlap",
    "purge_boundary",
    "next_period_first_target_baseline",
    "baseline_to_next_period_sessions",
    "target_end_to_next_period_sessions",
    "target_access_policy",
)


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def _raw_period(sample_date: date) -> str:
    if DEVELOPMENT_START <= sample_date <= DEVELOPMENT_END:
        return "development"
    if VALIDATION_START <= sample_date <= VALIDATION_END:
        return "validation"
    if FINAL_TEST_START <= sample_date <= FINAL_TEST_END:
        return "final_test"
    raise ValueError(
        "Sample date is outside the frozen chronological periods: "
        f"{sample_date.isoformat()}"
    )


def build_split_manifest(
    provenance: pd.DataFrame,
    canonical_sessions: Sequence[date],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign frozen periods and purge rows whose labels cross a boundary.

    Purging is based on canonical-session positions and each row's stored
    ``target_end_date``.  It deliberately does not remove an arbitrary number
    of event-aware rows.
    """
    required = {
        "sample_date",
        "target_baseline_date",
        "target_end_date",
    }
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Split provenance is missing columns: {missing}")
    if provenance["sample_date"].duplicated().any():
        raise ValueError("Split provenance contains duplicate sample dates.")

    sessions = tuple(canonical_sessions)
    positions = {session: position for position, session in enumerate(sessions)}
    if len(positions) != len(sessions):
        raise ValueError("Canonical trading sessions are not unique.")

    frame = provenance.loc[
        :, ["sample_date", "target_baseline_date", "target_end_date"]
    ].copy()
    frame["sample_date"] = frame["sample_date"].map(_as_date)
    frame["target_baseline_date"] = frame["target_baseline_date"].map(_as_date)
    frame["target_end_date"] = frame["target_end_date"].map(_as_date)
    frame["raw_period"] = frame["sample_date"].map(_raw_period)

    for column in ("target_baseline_date", "target_end_date"):
        absent = sorted(set(frame[column]) - set(positions))
        if absent:
            raise ValueError(
                f"{column} contains non-canonical dates: "
                f"{[value.isoformat() for value in absent[:5]]}"
            )

    validation_rows = frame[frame["raw_period"] == "validation"]
    final_test_rows = frame[frame["raw_period"] == "final_test"]
    if validation_rows.empty or final_test_rows.empty:
        raise ValueError("Validation and final-test periods must both contain rows.")

    validation_first_baseline = min(validation_rows["target_baseline_date"])
    final_test_first_baseline = min(final_test_rows["target_baseline_date"])
    if validation_first_baseline != VALIDATION_START:
        raise ValueError(
            "The first validation target baseline no longer matches the frozen "
            f"boundary: {validation_first_baseline}"
        )
    if final_test_first_baseline != FINAL_TEST_START:
        raise ValueError(
            "The first final-test target baseline no longer matches the frozen "
            f"boundary: {final_test_first_baseline}"
        )

    boundary_by_period = {
        "development": validation_first_baseline,
        "validation": final_test_first_baseline,
    }
    modeling_roles: list[str] = []
    eligible: list[bool] = []
    purged: list[bool] = []
    purge_boundaries: list[str] = []
    next_baselines: list[str] = []
    baseline_distances: list[int | None] = []
    endpoint_distances: list[int | None] = []
    access_policies: list[str] = []

    for row in frame.itertuples(index=False):
        period = row.raw_period
        if period == "final_test":
            modeling_roles.append("final_test_locked")
            eligible.append(True)
            purged.append(False)
            purge_boundaries.append("")
            next_baselines.append("")
            baseline_distances.append(None)
            endpoint_distances.append(None)
            access_policies.append("LOCKED_UNTIL_FINAL_EVALUATION")
            continue

        boundary = boundary_by_period[period]
        baseline_distance = (
            positions[boundary] - positions[row.target_baseline_date]
        )
        endpoint_distance = positions[boundary] - positions[row.target_end_date]
        crosses = row.target_end_date >= boundary

        purged.append(crosses)
        eligible.append(not crosses)
        purge_boundaries.append(
            "development_to_validation"
            if period == "development"
            else "validation_to_final_test"
        )
        next_baselines.append(boundary.isoformat())
        baseline_distances.append(int(baseline_distance))
        endpoint_distances.append(int(endpoint_distance))
        access_policies.append(
            "AVAILABLE_FOR_DEVELOPMENT"
            if period == "development"
            else "AVAILABLE_FOR_VALIDATION"
        )
        if crosses:
            modeling_roles.append(f"{period}_boundary_purge")
        elif period == "development":
            modeling_roles.append("development_train")
        else:
            modeling_roles.append("validation")

    frame["modeling_role"] = modeling_roles
    frame["eligible_for_period_evaluation"] = eligible
    frame["purged_for_target_overlap"] = purged
    frame["purge_boundary"] = purge_boundaries
    frame["next_period_first_target_baseline"] = next_baselines
    frame["baseline_to_next_period_sessions"] = pd.array(
        baseline_distances, dtype="Int64"
    )
    frame["target_end_to_next_period_sessions"] = pd.array(
        endpoint_distances, dtype="Int64"
    )
    frame["target_access_policy"] = access_policies

    retained_development = frame[
        frame["modeling_role"] == "development_train"
    ]
    retained_validation = frame[frame["modeling_role"] == "validation"]
    if retained_development.empty or retained_validation.empty:
        raise ValueError("Purging removed an entire modeling period.")

    last_development = retained_development.iloc[-1]
    last_validation = retained_validation.iloc[-1]
    development_gap = (
        positions[validation_first_baseline]
        - positions[last_development["target_baseline_date"]]
    )
    validation_gap = (
        positions[final_test_first_baseline]
        - positions[last_validation["target_baseline_date"]]
    )

    validations = {
        "periods_are_chronological_and_nonoverlapping": bool(
            DEVELOPMENT_END < VALIDATION_START
            <= VALIDATION_END
            < FINAL_TEST_START
        ),
        "all_target_dates_are_canonical_sessions": True,
        "development_labels_end_before_validation_baseline": bool(
            (
                retained_development["target_end_date"]
                < validation_first_baseline
            ).all()
        ),
        "validation_labels_end_before_final_test_baseline": bool(
            (
                retained_validation["target_end_date"]
                < final_test_first_baseline
            ).all()
        ),
        "development_boundary_gap_exceeds_21_sessions": development_gap > 21,
        "validation_boundary_gap_exceeds_21_sessions": validation_gap > 21,
        "purge_uses_target_end_dates_not_event_row_counts": True,
        "final_test_is_marked_locked": bool(
            (
                frame.loc[
                    frame["raw_period"] == "final_test",
                    "target_access_policy",
                ]
                == "LOCKED_UNTIL_FINAL_EVALUATION"
            ).all()
        ),
    }
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise ValueError(f"Chronological split validation failed: {failed}")

    for column in ("sample_date", "target_baseline_date", "target_end_date"):
        frame[column] = frame[column].map(date.isoformat)
    frame = frame.loc[:, list(SPLIT_MANIFEST_COLUMNS)]

    role_counts = {
        str(name): int(count)
        for name, count in frame["modeling_role"].value_counts().sort_index().items()
    }
    raw_period_counts = {
        str(name): int(count)
        for name, count in frame["raw_period"].value_counts().sort_index().items()
    }
    summary = {
        "boundary_source": (
            "Frozen Step 4 canonical-calendar 60/20/20 planning dates, now "
            "made final and protected with target-overlap-aware purging."
        ),
        "canonical_calendar_rule": (
            "Observed non-missing Yahoo ^GSPC Close sessions."
        ),
        "frozen_periods": {
            "development": {
                "start": DEVELOPMENT_START.isoformat(),
                "end": DEVELOPMENT_END.isoformat(),
            },
            "validation": {
                "start": VALIDATION_START.isoformat(),
                "end": VALIDATION_END.isoformat(),
            },
            "final_test": {
                "start": FINAL_TEST_START.isoformat(),
                "end": FINAL_TEST_END.isoformat(),
            },
        },
        "first_validation_target_baseline": (
            validation_first_baseline.isoformat()
        ),
        "first_final_test_target_baseline": (
            final_test_first_baseline.isoformat()
        ),
        "raw_period_counts": raw_period_counts,
        "modeling_role_counts": role_counts,
        "development_purged_rows": int(
            (frame["modeling_role"] == "development_boundary_purge").sum()
        ),
        "validation_purged_rows": int(
            (frame["modeling_role"] == "validation_boundary_purge").sum()
        ),
        "last_retained_development_sample": str(
            retained_development["sample_date"].iloc[-1]
        ),
        "last_retained_development_target_baseline": str(
            retained_development["target_baseline_date"].iloc[-1]
        ),
        "last_retained_development_target_end": str(
            retained_development["target_end_date"].iloc[-1]
        ),
        "last_retained_validation_sample": str(
            retained_validation["sample_date"].iloc[-1]
        ),
        "last_retained_validation_target_baseline": str(
            retained_validation["target_baseline_date"].iloc[-1]
        ),
        "last_retained_validation_target_end": str(
            retained_validation["target_end_date"].iloc[-1]
        ),
        "development_to_validation_baseline_gap_sessions": int(
            development_gap
        ),
        "validation_to_final_test_baseline_gap_sessions": int(validation_gap),
        "purge_rule": (
            "Retain a row in the earlier period only when its target_end_date "
            "is strictly before the next period's first target_baseline_date."
        ),
        "validations": validations,
    }
    return frame, summary
