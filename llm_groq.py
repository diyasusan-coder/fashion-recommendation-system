"""
Outfit-reasoning layer using Groq's free LLM API (Llama 3).

If GROQ_API_KEY isn't set or the request fails (e.g. no wifi during the demo),
this falls back to a decent templated suggestion so the app never breaks
mid-presentation.
"""

import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


def _fallback_suggestion(garment, color, occasion, season):
    return (
        f"For your {color} {garment}, try pairing it with neutral-toned "
        f"complements suited for a {occasion} in {season}. Clean lines and "
        f"minimal accessories will keep the look versatile — add one statement "
        f"piece (a bag, shoes, or jewelry) to make it pop."
    )


def get_outfit_recommendation(garment: str, color: str, occasion: str, season: str) -> dict:
    """
    Returns {"suggestion": str, "pairing_query": str, "source": "groq"|"fallback"}

    pairing_query is a short search phrase used to fetch real reference photos
    (e.g. "beige chinos white sneakers outfit") via the Pexels image search.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        return {
            "suggestion": _fallback_suggestion(garment, color, occasion, season),
            "pairing_query": f"{color} {garment} outfit {occasion}",
            "source": "fallback",
        }

    prompt = (
        f"A user is wearing a {color} {garment}. They want outfit advice for "
        f"a {occasion} during {season}. In 2-3 short sentences, suggest what "
        f"to pair it with (colors, specific items, accessories). Then on a "
        f"new line write exactly: SEARCH: <a short 4-6 word image search phrase "
        f"for the overall recommended outfit look>."
    )

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 220,
            },
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        pairing_query = f"{color} {garment} outfit {occasion}"
        suggestion_text = content
        if "SEARCH:" in content:
            suggestion_text, search_part = content.split("SEARCH:", 1)
            pairing_query = search_part.strip() or pairing_query
            suggestion_text = suggestion_text.strip()

        return {"suggestion": suggestion_text, "pairing_query": pairing_query, "source": "groq"}

    except Exception:
        return {
            "suggestion": _fallback_suggestion(garment, color, occasion, season),
            "pairing_query": f"{color} {garment} outfit {occasion}",
            "source": "fallback",
        }
