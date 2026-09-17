from summary_validator import build_fallback_summary, validate_summary


def base_market_state(stress_flags=None):
    if stress_flags is None:
        stress_flags = []

    return {
        "as_of": "2026-07-04",
        "regime": {"label": "risk_on", "confidence": 0.7},
        "drivers": [
            {
                "name": "sp500_5d_return",
                "value": 0.4,
                "unit": "pct",
                "evidence_id": "sp500_5d_return",
            },
            {
                "name": "nasdaq_5d_return",
                "value": 0.6,
                "unit": "pct",
                "evidence_id": "nasdaq_5d_return",
            },
        ],
        "asset_class_snapshot": [
            {
                "class": "equities",
                "members": [
                    {
                        "asset": "sp500",
                        "metric": "sp500_5d_return",
                        "return_5d_pct": 0.4,
                        "evidence_id": "sp500_5d_return",
                    },
                    {
                        "asset": "nasdaq",
                        "metric": "nasdaq_5d_return",
                        "return_5d_pct": 0.6,
                        "evidence_id": "nasdaq_5d_return",
                    },
                ],
            }
        ],
        "stress_flags": stress_flags,
        "mood_5d": {
            "label": "bullish",
            "prob": 0.67,
            "top_features": [
                {"name": "sp500_5d_return", "contribution": 0.004},
            ],
        },
        "evidence": {
            "sp500_5d_return": {"display_value": 0.4, "unit": "pct"},
            "nasdaq_5d_return": {"display_value": 0.6, "unit": "pct"},
            "vix_20d_volatility": {
                "display_value": 28.5,
                "unit": "pct_annualized",
            },
        },
    }


def valid_summary(risks_to_watch=None):
    if risks_to_watch is None:
        risks_to_watch = []

    return {
        "headline": "Markets are in a risk-on regime.",
        "regime_label": "risk_on",
        "regime_summary": {
            "text": "S&P 500 5d return is +0.4%.",
            "evidence_ids": ["sp500_5d_return"],
        },
        "key_changes": [
            {
                "text": "Nasdaq 5d return is +0.6%.",
                "evidence_ids": ["nasdaq_5d_return"],
            }
        ],
        "risks_to_watch": risks_to_watch,
        "mood_5d": {
            "label": "bullish",
            "probability": 0.67,
            "drivers": [
                {
                    "text": "S&P 500 helped the 5d mood.",
                    "evidence_ids": ["sp500_5d_return"],
                }
            ],
        },
        "limitations": [],
    }


def test_valid_cited_summary_passes():
    result = validate_summary(base_market_state(), valid_summary())

    assert result.passed
    assert result.errors == []


def test_mood_mismatch_fails():
    summary = valid_summary()
    summary["mood_5d"]["label"] = "bearish"

    result = validate_summary(base_market_state(), summary)

    assert not result.passed
    assert any("mood_5d.label mismatch" in e for e in result.errors)


def test_unknown_evidence_id_fails():
    summary = valid_summary()
    summary["key_changes"][0]["evidence_ids"] = ["unknown_metric"]

    result = validate_summary(base_market_state(), summary)

    assert not result.passed
    assert any("unknown evidence_id" in e for e in result.errors)


def test_risk_without_active_stress_flag_fails():
    summary = valid_summary(
        risks_to_watch=[
            {
                "text": "Volatility stress is elevated at 28.5%.",
                "evidence_ids": ["vix_20d_volatility"],
            }
        ]
    )

    result = validate_summary(base_market_state(), summary)

    assert not result.passed
    assert "risks_to_watch must be empty when no stress_flags are active" in result.errors


def test_signed_numeric_mismatch_fails():
    summary = valid_summary()
    summary["regime_summary"]["text"] = "S&P 500 5d return is -0.4%."

    result = validate_summary(base_market_state(), summary)

    assert not result.passed
    assert any("does not match cited evidence" in e for e in result.errors)


def test_empty_regime_summary_fails():
    summary = valid_summary()
    summary["regime_summary"] = {"text": "", "evidence_ids": []}

    result = validate_summary(base_market_state(), summary)

    assert not result.passed
    assert "regime_summary is missing or empty" in result.errors


def test_fallback_summary_passes_validation():
    fallback = build_fallback_summary(base_market_state())

    result = validate_summary(base_market_state(), fallback)

    assert result.passed
    assert result.errors == []
