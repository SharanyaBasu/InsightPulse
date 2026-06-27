import os
from dotenv import load_dotenv
import json
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "regime_summary": {
            "type": "array",
            "items": {"type": "string"},
        },
        "key_changes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "risks_to_watch": {
            "type": "array",
            "items": {"type": "string"},
        },
        "mood_5d": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "probability": {"type": "number"},
                "drivers": {
                    "type": "array",
                    "items": {"type": "string"},
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

Guidelines:
- No investment advice (avoid buy, sell, should, recommend).
- Write a neutral, descriptive daily summary grounded in regime, drivers,
  asset_class_snapshot, stress_flags, and mood_5d.
- Include specific numbers from drivers where relevant.
- Set mood_5d.probability from mood_5d.prob in the input.
- Only list risks present in stress_flags; leave risks_to_watch empty if none are active.
- correlation_shifts may be empty — do not fabricate correlation narratives.
"""


def generate_summary(market_state: dict) -> dict:
    prompt = f"""{SYSTEM_PROMPT}

    MarketState JSON:
    {json.dumps(market_state, indent=2)}
    """

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
