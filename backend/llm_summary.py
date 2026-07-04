import os
from dotenv import load_dotenv
import json

from summary_validator import build_fallback_summary, validate_summary

load_dotenv()

CITED_BULLET_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["text", "evidence_ids"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "regime_summary": CITED_BULLET_SCHEMA,
        "key_changes": {
            "type": "array",
            "items": CITED_BULLET_SCHEMA,
        },
        "risks_to_watch": {
            "type": "array",
            "items": CITED_BULLET_SCHEMA,
        },
        "mood_5d": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "probability": {"type": "number"},
                "drivers": {
                    "type": "array",
                    "items": CITED_BULLET_SCHEMA,
                },
            },
            "required": ["label", "probability", "drivers"],
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "headline",
        "regime_summary",
        "key_changes",
        "risks_to_watch",
        "mood_5d",
        "limitations",
    ],
}

SYSTEM_PROMPT = """You are InsightPulse's market narrative generator.

Rules:
- Use ONLY the information in the provided MarketState JSON.
- Do NOT invent events, news, or macro explanations not present.
- If evidence is mixed or weak, mention uncertainty in "limitations".
- Return ONLY valid JSON matching the schema.

Citation rules (regime_summary, key_changes, risks_to_watch, mood_5d.drivers):
- regime_summary is a single object with "text" and "evidence_ids" (not an array).
- Write regime_summary.text as one concise paragraph (1–3 sentences) summarizing the regime backdrop.
- Include all supporting evidence_ids in regime_summary.evidence_ids (at least one valid ID).
- key_changes, risks_to_watch, mood_5d.drivers: each bullet is an object with "text" and "evidence_ids".
- Each bullet's evidence_ids array MUST contain at least one valid evidence ID.
- Every evidence_id MUST be a key in MarketState.evidence — do not invent IDs.
- regime_summary: cite drivers, regime-related metrics, or specific asset_class_snapshot member evidence_id values.
- key_changes bullets: cite specific asset_class_snapshot member evidence_id values or relevant driver evidence_ids.
- risks_to_watch bullets: cite ONLY evidence_ids from active stress_flags.
- If no stress_flags are active, risks_to_watch MUST be [] with no bullet objects.
- mood_5d.drivers bullets: cite mood_5d.top_features metric names (they exist in evidence).
- If a claim cannot be tied to a MarketState.evidence key, omit it rather than using empty evidence_ids.
- Include specific numbers from evidence display_value where relevant in the text.
- Do NOT describe asset-class averages or aggregate returns unless they are explicit evidence keys.
- Do NOT quote internal scoring fields such as regime.confidence, driver z-scores, or mood feature contributions.

Guidelines:
- No investment advice (avoid buy, sell, should, recommend).
- Write a neutral, descriptive daily summary grounded in regime, drivers,
  asset_class_snapshot, stress_flags, and mood_5d.
- Set mood_5d.label and mood_5d.probability from mood_5d.label and mood_5d.prob in the input.
- Only list risks present in stress_flags; leave risks_to_watch empty if none are active.
- correlation_shifts may be empty — do not fabricate correlation narratives.
- headline and limitations remain plain strings (no evidence_ids on those fields).
"""


def _get_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from backend/.env")

    try:
        import google.generativeai as genai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Run `pip install -r requirements.txt` "
            "from the backend virtual environment."
        ) from exc

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def generate_summary(
    market_state: dict,
    *,
    validation_feedback: list[str] | None = None,
) -> dict:
    feedback_block = ""
    if validation_feedback:
        feedback_lines = "\n".join(f"- {error}" for error in validation_feedback)
        feedback_block = f"""

    Your previous output failed validation. Fix ONLY the unsupported claims.
    Validation errors:
    {feedback_lines}
    """

    prompt = f"""{SYSTEM_PROMPT}{feedback_block}

    MarketState JSON:
    {json.dumps(market_state, indent=2)}
    """

    model = _get_model()
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": SUMMARY_SCHEMA,
        },
    )

    parsed = json.loads(response.text)
    parsed["as_of"] = market_state.get("as_of")
    parsed["regime_label"] = market_state.get("regime", {}).get("label")
    return parsed


def generate_summary_with_guardrail(
    market_state: dict,
    max_retries: int = 1,
) -> dict:
    attempts = max_retries + 1
    last_errors: list[str] = []

    for attempt in range(attempts):
        validation_feedback = last_errors if attempt > 0 else None
        summary = generate_summary(
            market_state,
            validation_feedback=validation_feedback,
        )
        result = validate_summary(market_state, summary)
        if result.passed:
            summary["validation"] = {"status": "verified"}
            return summary

        last_errors = result.errors
        print(
            f"Summary validation failed (attempt {attempt + 1}/{attempts}):",
            last_errors,
        )

    print("Summary validation failed after retries; using fallback.")
    fallback = build_fallback_summary(market_state)
    fallback_result = validate_summary(market_state, fallback)
    if not fallback_result.passed:
        print("WARNING: fallback summary failed validation:", fallback_result.errors)
    fallback["validation"] = {"status": "fallback", "errors": last_errors}
    return fallback
