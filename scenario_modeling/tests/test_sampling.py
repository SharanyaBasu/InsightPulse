from __future__ import annotations

import json
import unittest
from datetime import date, timedelta

import pandas as pd

from scenario_modeling.src.sampling import (
    EVENT_AWARE_INDEX_COLUMNS,
    EVENT_TYPE_RANK,
    SHARED_START,
    TARGET_HORIZON_SESSIONS,
    TARGET_SPECS,
    WEEKLY_INDEX_COLUMNS,
    _event_record,
    _frame_to_csv_text,
    _label_metadata,
    build_sampling_artifacts,
    construct_event_aware_candidate,
    next_observed_session,
    nth_future_session,
    weekly_anchors,
)


def _weekday_sessions(start: date, count: int) -> tuple[date, ...]:
    sessions: list[date] = []
    current = start
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    return tuple(sessions)


class SamplingUnitTests(unittest.TestCase):
    def test_weekly_anchor_uses_thursday_for_good_friday(self) -> None:
        sessions = (
            date(2024, 3, 25),
            date(2024, 3, 26),
            date(2024, 3, 27),
            date(2024, 3, 28),
            date(2024, 4, 1),
            date(2024, 4, 2),
            date(2024, 4, 3),
            date(2024, 4, 4),
            date(2024, 4, 5),
        )
        anchors = weekly_anchors(
            sessions, date(2024, 3, 25), date(2024, 4, 5)
        )
        self.assertEqual(anchors[0]["anchor_date"], date(2024, 3, 28))
        self.assertTrue(anchors[0]["anchor_is_shifted"])
        self.assertEqual(
            anchors[0]["anchor_shift_reason"], "friday_market_holiday"
        )
        self.assertEqual(anchors[1]["anchor_date"], date(2024, 4, 5))
        self.assertFalse(anchors[1]["anchor_is_shifted"])

    def test_next_session_shift_is_strict(self) -> None:
        sessions = (
            date(2024, 1, 5),
            date(2024, 1, 8),
            date(2024, 1, 9),
        )
        self.assertEqual(
            next_observed_session(
                sessions, date(2024, 1, 5), strictly_after=True
            ),
            date(2024, 1, 8),
        )
        self.assertEqual(
            next_observed_session(
                sessions, date(2024, 1, 6), strictly_after=False
            ),
            date(2024, 1, 8),
        )

    def test_unknown_and_after_close_event_timing(self) -> None:
        sessions = _weekday_sessions(date(2024, 1, 2), 10)
        unknown = _event_record(
            event_id="cpi:test",
            event_type="cpi",
            event_source="test",
            event_source_file="test.json",
            observation_period="2023-12-01",
            release_date=date(2024, 1, 4),
            release_time=None,
            release_timezone=None,
            release_date_kind="ALFRED_INITIAL_APPEARANCE",
            source_model_availability_date=None,
            availability_rule="unknown time",
            conservative_next_session_shift=True,
            sessions=sessions,
        )
        self.assertEqual(
            unknown["state_cutoff_date"], date(2024, 1, 3)
        )
        self.assertEqual(
            unknown["model_availability_date"], date(2024, 1, 5)
        )

        after_close = _event_record(
            event_id="fomc:test",
            event_type="fomc",
            event_source="test",
            event_source_file="test.json",
            observation_period="test",
            release_date=date(2024, 1, 4),
            release_time="17:00:00",
            release_timezone="ET",
            release_date_kind="official_fomc_announcement",
            source_model_availability_date=date(2024, 1, 5),
            availability_rule="after close",
            conservative_next_session_shift=False,
            sessions=sessions,
        )
        self.assertEqual(
            after_close["state_cutoff_date"], date(2024, 1, 4)
        )
        self.assertEqual(
            after_close["model_availability_date"], date(2024, 1, 5)
        )

    def test_21_session_boundary_is_exact(self) -> None:
        sessions = _weekday_sessions(date(2024, 1, 2), 30)
        availability = {
            spec.target: set(sessions) for spec in TARGET_SPECS
        }
        baseline = sessions[8]
        metadata = _label_metadata(
            baseline,
            sessions=sessions,
            shared_end=sessions[-1],
            target_availability=availability,
        )
        self.assertEqual(
            metadata["target_end_date"],
            nth_future_session(
                sessions, baseline, TARGET_HORIZON_SESSIONS
            ).isoformat(),
        )
        self.assertTrue(metadata["label_eligible"])

        boundary = _label_metadata(
            sessions[9],
            sessions=sessions,
            shared_end=sessions[-1],
            target_availability=availability,
        )
        self.assertFalse(boundary["has_21_future_sessions"])
        self.assertFalse(boundary["label_eligible"])

    def test_same_day_events_are_deduplicated_and_preserved(self) -> None:
        sessions = _weekday_sessions(date(2024, 1, 2), 40)
        availability = {
            spec.target: set(sessions) for spec in TARGET_SPECS
        }
        event_date = sessions[5]
        records = []
        for event_type, event_id in (("cpi", "cpi:one"), ("fomc", "fomc:one")):
            records.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "event_source": "test",
                    "event_observation_period": "test",
                    "original_release_date": sessions[4].isoformat(),
                    "original_release_time": "",
                    "original_release_timezone": "",
                    "release_date_kind": "test",
                    "model_availability_date": event_date.isoformat(),
                    "next_session_shift_applied": True,
                    "conservative_next_session_shift": True,
                    "calendar_normalization_applied": False,
                    "state_cutoff_date": sessions[3].isoformat(),
                }
            )
        events = pd.DataFrame(records)
        candidate = construct_event_aware_candidate(
            sessions=sessions,
            events=events,
            target_availability=availability,
            shared_end=sessions[-1],
        )
        self.assertEqual(len(candidate), 1)
        self.assertEqual(candidate.iloc[0]["event_count"], 2)
        self.assertEqual(candidate.iloc[0]["duplicate_candidates_removed"], 1)
        self.assertEqual(
            json.loads(candidate.iloc[0]["event_types_json"]),
            sorted(["cpi", "fomc"], key=EVENT_TYPE_RANK.get),
        )
        self.assertCountEqual(
            json.loads(candidate.iloc[0]["event_ids_json"]),
            ["cpi:one", "fomc:one"],
        )


class CachedSamplingAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = build_sampling_artifacts()

    def test_actual_cached_counts_and_selected_design(self) -> None:
        summary = self.artifacts["summary"]
        self.assertEqual(summary["construction_mode"], "offline_cached_data_only")
        self.assertEqual(summary["selected_design"], "EVENT_AWARE")
        self.assertEqual(summary["weekly"]["raw_rows"], 677)
        self.assertEqual(summary["weekly"]["label_eligible_rows"], 654)
        self.assertEqual(summary["event_aware"]["raw_rows"], 591)
        self.assertEqual(summary["event_aware"]["label_eligible_rows"], 585)
        self.assertEqual(
            summary["event_aware"]["total_events_before_deduplication"], 621
        )

    def test_actual_release_dates_are_not_observation_months(self) -> None:
        events = self.artifacts["normalized_events"]
        macro = events[
            events["event_type"].isin(
                ["cpi", "unemployment", "gdp", "pmi"]
            )
        ]
        self.assertTrue(
            (
                macro["original_release_date"]
                != macro["event_observation_period"]
            ).all()
        )
        self.assertTrue(
            self.artifacts["summary"]["point_in_time_validation"][
                "no_observation_month_used_as_release_date"
            ]
        )

    def test_no_duplicate_selected_timestamp_and_all_events_preserved(self) -> None:
        selected = self.artifacts["selected"]
        self.assertFalse(selected["sample_date"].duplicated().any())
        collisions = self.artifacts["summary"]["event_aware"][
            "same_day_collisions"
        ]
        self.assertEqual(collisions["event_identity_loss_count"], 0)
        self.assertEqual(collisions["same_day_collision_dates"], 30)

    def test_stable_schemas_ordering_and_repeatable_serialization(self) -> None:
        weekly = self.artifacts["weekly"]
        event_aware = self.artifacts["event_aware"]
        self.assertEqual(tuple(weekly.columns), WEEKLY_INDEX_COLUMNS)
        self.assertEqual(
            tuple(event_aware.columns), EVENT_AWARE_INDEX_COLUMNS
        )
        self.assertTrue(weekly["sample_date"].is_monotonic_increasing)
        self.assertTrue(event_aware["sample_date"].is_monotonic_increasing)
        self.assertEqual(
            _frame_to_csv_text(event_aware),
            _frame_to_csv_text(event_aware.copy()),
        )

    def test_shared_period_and_exact_horizon(self) -> None:
        summary = self.artifacts["summary"]
        self.assertEqual(summary["shared_period"]["start"], SHARED_START.isoformat())
        self.assertEqual(summary["shared_period"]["end"], "2026-07-16")
        self.assertTrue(
            summary["point_in_time_validation"][
                "all_nonblank_target_ends_are_exactly_21_sessions"
            ]
        )


if __name__ == "__main__":
    unittest.main()
