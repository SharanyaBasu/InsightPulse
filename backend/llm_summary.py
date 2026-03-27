import os
from dotenv import load_dotenv
import json
import google.generativeai as genai

load_dotenv()

# Configure Gemini API key
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
                }
            },
            "required": ["label", "probability", "drivers"]
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": [
        "headline",
        "regime_summary",
        "key_changes",
        "risks_to_watch",
        "mood_5d",
        "limitations"
    ]
}

def generate_summary(market_state: dict) -> dict:
    prompt = f"""
    You are InsightPulse's market narrative generator.

    Rules:
    - Use ONLY the information in the provided MarketState JSON.
    - Do NOT invent events, news, or macro explanations not present.
    - If evidence is mixed or weak, mention uncertainty in "limitations".
    - Return ONLY valid JSON matching the schema.
    - For array fields like "regime_summary", "key_changes", "risks_to_watch", "drivers" and "limitations":
        - Make sure "regime_summary" has 2 to 4 items.
        - Make sure "key_changes" has 2 to 5 items.
        - Make sure "risks_to_watch" has 1 to 3 items.
        - Make sure "drivers" has 1 to 3 items.
        - Make sure "limitations" has 1 to 3 items.

    MarketState JSON:
    {json.dumps(market_state, indent=2)}
    """

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": SUMMARY_SCHEMA
        }
    )

    return json.loads(response.text)