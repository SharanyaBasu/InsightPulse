"""Deterministic validation and fallback generation for LLM market summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MOOD_PROB_TOLERANCE = 0.01
NUMERIC_TOLERANCE = 0.25


@dataclass
class ValidationResult:
    """Store the result of summary validation."""

    passed: bool
    errors: list[str] = field(default_factory=list)


def normalize_bullet(item) -> dict | None:
    """Accept legacy string bullets or cited {text, evidence_ids} objects."""
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {"text": text, "evidence_ids": [], "legacy": True}

    if isinstance(item, dict):
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        evidence_ids = item.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        return {
            "text": text.strip(),
            "evidence_ids": [eid for eid in evidence_ids if isinstance(eid, str) and eid],
            "legacy": False,
        }

    return None


def _normalize_bullet_list(items) -> list[dict]:
    """Normalize a list of summary bullets."""

    if not isinstance(items, list):
        return []
    bullets = []
    for item in items:
        normalized = normalize_bullet(item)
        if normalized:
            bullets.append(normalized)
    return bullets


def _normalize_cited_value(item) -> list[dict]:
    """Regime summary: cited object, legacy string, or legacy bullet array."""
    if isinstance(item, list):
        return _normalize_bullet_list(item)

    normalized = normalize_bullet(item)
    if not normalized:
        return []
    return [normalized]


def _active_stress_flag_evidence_ids(market_state: dict) -> set[str]:
    """Return evidence IDs for active stress flags."""

    ids: set[str] = set()
    for flag in market_state.get("stress_flags") or []:
        if not flag.get("active", True):
            continue
        eid = flag.get("evidence_id")
        if isinstance(eid, str) and eid:
            ids.add(eid)
    return ids


def _active_stress_flag_names(market_state: dict) -> set[str]:
    """Return names of active stress flags."""

    names: set[str] = set()
    for flag in market_state.get("stress_flags") or []:
        if not flag.get("active", True):
            continue
        name = flag.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _extract_numbers(text: str) -> list[float]:
    """Extract unique percentages and decimal values from text."""

    numbers: list[float] = []
    seen: set[float] = set()

    for match in re.finditer(r"([-+]?\d+\.?\d*)\s*%", text):
        value = float(match.group(1))
        if value not in seen:
            numbers.append(value)
            seen.add(value)

    for match in re.finditer(r"(?<!\d)([-+]?\d+\.\d+)(?!\d)", text):
        value = float(match.group(1))
        if value not in seen:
            numbers.append(value)
            seen.add(value)

    return numbers


def _number_matches_cited_evidence(
    number: float,
    evidence_ids: list[str],
    evidence: dict,
) -> bool:
    """Check whether a number matches any cited evidence value."""

    for eid in evidence_ids:
        row = evidence.get(eid)
        if not row:
            continue
        display = row.get("display_value")
        if display is None:
            continue
        display_value = float(display)
        if abs(number - display_value) <= NUMERIC_TOLERANCE:
            return True
    return False


def _validate_cited_bullets(
    field_name: str,
    bullets: list[dict],
    evidence: dict,
    errors: list[str],
    *,
    allowed_evidence_ids: set[str] | None = None,
) -> None:
    """Validate citations and numeric claims in summary bullets.

    Args:
        field_name: Summary field being validated.
        bullets: Normalized bullets to validate.
        evidence: Evidence records keyed by metric ID.
        errors: List that receives validation errors.
        allowed_evidence_ids: Optional set of evidence IDs allowed for the field.
    """

    for index, bullet in enumerate(bullets):
        prefix = f"{field_name}[{index}]"

        if bullet.get("legacy"):
            continue

        evidence_ids = bullet["evidence_ids"]
        if not evidence_ids:
            errors.append(f"{prefix}: cited bullet missing evidence_ids")
            continue

        for eid in evidence_ids:
            if eid not in evidence:
                errors.append(f"{prefix}: unknown evidence_id '{eid}'")
            elif allowed_evidence_ids is not None and eid not in allowed_evidence_ids:
                errors.append(
                    f"{prefix}: evidence_id '{eid}' is not an active stress flag"
                )

        for number in _extract_numbers(bullet["text"]):
            if not _number_matches_cited_evidence(number, evidence_ids, evidence):
                errors.append(
                    f"{prefix}: number {number} in text does not match cited evidence"
                )


def _validate_legacy_risk_bullets(
    bullets: list[dict],
    allowed_names: set[str],
    errors: list[str],
) -> None:
    """Check that legacy risk bullets reference active stress flags."""

    for index, bullet in enumerate(bullets):
        if not bullet.get("legacy"):
            continue
        text = bullet["text"].lower()
        if allowed_names and not any(name in text for name in allowed_names):
            errors.append(
                f"risks_to_watch[{index}]: legacy risk bullet does not reference an active stress flag"
            )


def validate_summary(market_state: dict, summary: dict) -> ValidationResult:
    """Validate a generated summary against the market state.

    Args:
        market_state: Structured market data used as the source of truth.
        summary: Generated summary to validate.

    Returns:
        The validation result with any errors found.
    """

    errors: list[str] = []

    if not isinstance(summary, dict):
        return ValidationResult(passed=False, errors=["summary must be an object"])

    evidence = market_state.get("evidence") or {}
    input_mood = market_state.get("mood_5d") or {}
    input_regime_label = (market_state.get("regime") or {}).get("label")
    summary_mood = summary.get("mood_5d") or {}

    expected_label = input_mood.get("label")
    actual_label = summary_mood.get("label")
    if expected_label and actual_label != expected_label:
        errors.append(
            f"mood_5d.label mismatch: expected '{expected_label}', got '{actual_label}'"
        )

    expected_prob = input_mood.get("prob")
    actual_prob = summary_mood.get("probability")
    if expected_prob is not None and actual_prob is not None:
        if abs(float(actual_prob) - float(expected_prob)) > MOOD_PROB_TOLERANCE:
            errors.append(
                "mood_5d.probability mismatch: "
                f"expected {expected_prob}, got {actual_prob}"
            )
    elif expected_prob is not None and actual_prob is None:
        errors.append("mood_5d.probability is missing")

    expected_regime = summary.get("regime_label", input_regime_label)
    if input_regime_label and expected_regime != input_regime_label:
        errors.append(
            f"regime_label mismatch: expected '{input_regime_label}', got '{expected_regime}'"
        )

    stress_flag_evidence_ids = _active_stress_flag_evidence_ids(market_state)
    stress_flag_names = _active_stress_flag_names(market_state)
    risk_bullets = _normalize_bullet_list(summary.get("risks_to_watch"))

    if not stress_flag_evidence_ids and risk_bullets:
        errors.append("risks_to_watch must be empty when no stress_flags are active")

    regime_bullets = _normalize_cited_value(summary.get("regime_summary"))
    if not regime_bullets:
        errors.append("regime_summary is missing or empty")
    else:
        _validate_cited_bullets("regime_summary", regime_bullets, evidence, errors)

    _validate_cited_bullets(
        "key_changes",
        _normalize_bullet_list(summary.get("key_changes")),
        evidence,
        errors,
    )
    _validate_cited_bullets(
        "risks_to_watch",
        risk_bullets,
        evidence,
        errors,
        allowed_evidence_ids=stress_flag_evidence_ids if stress_flag_evidence_ids else None,
    )
    _validate_legacy_risk_bullets(risk_bullets, stress_flag_names, errors)

    mood_drivers = _normalize_bullet_list(summary_mood.get("drivers"))
    _validate_cited_bullets("mood_5d.drivers", mood_drivers, evidence, errors)

    return ValidationResult(passed=not errors, errors=errors)


def _format_driver_clause(driver: dict, evidence: dict) -> tuple[str, str | None]:
    """Format a market driver and return its evidence ID."""

    name = driver.get("name", "unknown")
    eid = driver.get("evidence_id")
    row = evidence.get(eid) if isinstance(eid, str) else None
    if row and row.get("display_value") is not None:
        value = row["display_value"]
        unit = row.get("unit") or driver.get("unit", "")
    else:
        value = driver.get("value")
        unit = driver.get("unit", "")

    if value is None:
        return name, eid if isinstance(eid, str) else None

    suffix = f" {unit}" if unit else ""
    return f"{name}: {value}{suffix}", eid if isinstance(eid, str) else None


def _format_asset_class_bullet(row: dict) -> list[str]:
    """Format asset class members as summary bullets."""

    asset_class = row.get("class", "unknown")
    bullets = []
    for member in row.get("members", []):
        metric = member.get("metric") or member.get("evidence_id")
        return_pct = member.get("return_5d_pct")
        if metric and return_pct is not None:
            bullets.append(f"{asset_class} {metric}: {return_pct}%")
    return bullets


def _format_stress_flag_bullet(flag: dict) -> str:
    """Format an active stress flag as a summary bullet."""

    name = flag.get("name", "unknown")
    severity = flag.get("severity")
    if severity:
        return f"{name} ({severity})"
    return name


def _format_mood_driver_bullet(feature: dict) -> str:
    """Format a mood feature as a summary bullet."""

    name = feature.get("name", "unknown")
    contribution = feature.get("contribution")
    if contribution is None:
        return name
    return f"{name} (contribution {contribution})"


def _build_regime_summary(market_state: dict) -> dict | str:
    """Build a regime summary from structured market data.

    Args:
        market_state: Structured market data used to build the summary.

    Returns:
        A cited regime summary, or plain text when no drivers are available.
    """

    regime_label = (market_state.get("regime") or {}).get("label", "unknown")
    drivers = market_state.get("drivers") or []
    evidence = market_state.get("evidence") or {}

    if not drivers:
        return f"The market is classified in a {regime_label} regime."

    evidence_ids: list[str] = []
    clauses: list[str] = []
    for driver in drivers[:3]:
        clause, eid = _format_driver_clause(driver, evidence)
        clauses.append(clause)
        if eid:
            evidence_ids.append(eid)

    text = f"The market is in a {regime_label} regime. " + " ".join(clauses)
    return {"text": text.strip(), "evidence_ids": evidence_ids}


def build_fallback_summary(market_state: dict) -> dict:
    """Build a deterministic summary when LLM validation fails.

    Args:
        market_state: Structured market data used to build the summary.

    Returns:
        A summary generated directly from the market state.
    """

    regime_label = (market_state.get("regime") or {}).get("label", "unknown")
    as_of = market_state.get("as_of", "")
    mood = market_state.get("mood_5d") or {}

    asset_snapshot = market_state.get("asset_class_snapshot") or []
    key_changes = [
        bullet
        for row in asset_snapshot
        for bullet in _format_asset_class_bullet(row)
    ]

    stress_flags = [
        flag for flag in (market_state.get("stress_flags") or [])
        if flag.get("active", True)
    ]
    risks_to_watch = [_format_stress_flag_bullet(flag) for flag in stress_flags]

    top_features = mood.get("top_features") or []
    mood_drivers = [_format_mood_driver_bullet(feature) for feature in top_features]

    return {
        "headline": f"Markets in {regime_label} regime as of {as_of}",
        "regime_summary": _build_regime_summary(market_state),
        "key_changes": key_changes,
        "risks_to_watch": risks_to_watch,
        "mood_5d": {
            "label": mood.get("label", "neutral"),
            "probability": mood.get("prob", 0.5),
            "drivers": mood_drivers,
        },
        "limitations": ["Generated from structured market data."],
        "as_of": as_of,
        "regime_label": regime_label,
    }
