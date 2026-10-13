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

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["passed", "issues"],
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

VERIFIER_PROMPT = """You are InsightPulse's market summary accuracy checker.

You will receive:
1. MarketState JSON (source of truth)
2. A generated market summary

Check whether the summary is accurate given the MarketState values.
Focus on factual accuracy: wrong numbers, wrong direction, invented causes,
unsupported claims, or risks that are not in active stress_flags.

Do NOT re-check citation ID formatting. That was already validated.
Return passed=true if the summary is accurate enough.
Return passed=false with short issue strings if it is not.
"""


def _get_model():
    """Create the configured Gemini model."""

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
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    return genai.GenerativeModel(model_name)


def generate_summary(
    market_state: dict,
    *,
    validation_feedback: list[str] | None = None,
) -> dict:
    """Generate a cited market summary from the market state.

    Args:
        market_state: Structured market data provided to the model.
        validation_feedback: Errors from a previous attempt, if any.

    Returns:
        The parsed summary with its date and regime label.
    """

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


def verify_summary_with_llm(summary: dict, market_state: dict) -> dict:
    """Check summary accuracy against market_state with a second LLM call.

    Args:
        summary: Summary that already passed deterministic validation.
        market_state: Structured market data used as the source of truth.

    Returns:
        Dict with passed (bool), issues (list of strings), and error (bool).
        error=True means the verifier call itself failed.
    """

    prompt = f"""{VERIFIER_PROMPT}

MarketState JSON:
{json.dumps(market_state, indent=2)}

Generated summary JSON:
{json.dumps(summary, indent=2)}
"""

    try:
        model = _get_model()
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": VERIFIER_SCHEMA,
            },
        )
        parsed = json.loads(response.text)
        return {
            "passed": bool(parsed.get("passed")),
            "issues": [
                issue for issue in (parsed.get("issues") or [])
                if isinstance(issue, str) and issue.strip()
            ],
            "error": False,
        }
    except Exception as exc:
        print("WARNING: semantic verifier failed:", exc)
        return {
            "passed": False,
            "issues": ["Semantic verifier could not complete the accuracy check."],
            "error": True,
        }


def _try_generate_valid_summary(
    market_state: dict,
    *,
    max_retries: int = 1,
    validation_feedback: list[str] | None = None,
) -> tuple[dict | None, list[str]]:
    """Generate until deterministic validation passes, or return None."""

    last_errors: list[str] = []
    feedback = validation_feedback

    for attempt in range(max_retries + 1):
        summary = generate_summary(
            market_state,
            validation_feedback=feedback,
        )
        result = validate_summary(market_state, summary)
        if result.passed:
            return summary, []

        last_errors = result.errors
        feedback = last_errors
        print(
            f"Summary validation failed (attempt {attempt + 1}/{max_retries + 1}):",
            last_errors,
        )

    return None, last_errors


def generate_summary_with_guardrail(
    market_state: dict,
    max_retries: int = 1,
) -> dict:
    """Generate, deterministically validate, then semantically verify a summary.

    Args:
        market_state: Structured market data used for generation and validation.
        max_retries: Number of retries after the first failed deterministic attempt.

    Returns:
        A verified LLM summary, an unverified LLM summary, or a data fallback.
    """

    summary, last_errors = _try_generate_valid_summary(
        market_state,
        max_retries=max_retries,
    )
    if summary is None:
        print("Summary validation failed after retries; using fallback.")
        fallback = build_fallback_summary(market_state)
        fallback_result = validate_summary(market_state, fallback)
        if not fallback_result.passed:
            print("WARNING: fallback summary failed validation:", fallback_result.errors)
        fallback["validation"] = {"status": "fallback", "errors": last_errors}
        return fallback

    semantic = verify_summary_with_llm(summary, market_state)
    if semantic["passed"]:
        summary["validation"] = {"status": "verified"}
        return summary

    # Verifier call itself failed — return summary with a warning, do not regenerate.
    if semantic.get("error"):
        print("Semantic verifier unavailable; returning unverified summary.")
        summary["validation"] = {
            "status": "unverified",
            "semantic_issues": semantic["issues"],
        }
        return summary

    print("Semantic verification failed:", semantic["issues"])
    retry_summary, _ = _try_generate_valid_summary(
        market_state,
        max_retries=0,
        validation_feedback=semantic["issues"],
    )
    if retry_summary is not None:
        retry_semantic = verify_summary_with_llm(retry_summary, market_state)
        if retry_semantic["passed"]:
            retry_summary["validation"] = {"status": "verified"}
            return retry_summary
        summary = retry_summary
        semantic = retry_semantic
        print("Semantic verification failed after regenerate:", semantic["issues"])

    summary["validation"] = {
        "status": "unverified",
        "semantic_issues": semantic["issues"],
    }
    return summary
