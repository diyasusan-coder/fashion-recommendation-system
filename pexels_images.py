"""
Fetches real reference photos for a pairing suggestion via the free Pexels API.
Falls back to an empty list (frontend shows a friendly placeholder) if no key
is set or the request fails.
"""

import os
import requests

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"


def search_pairing_images(query: str, per_page: int = 4) -> list:
    api_key = os.getenv("PEXELS_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        resp = requests.get(
            PEXELS_SEARCH_URL,
            headers={"Authorization": api_key},
            params={"query": query, "per_page": per_page, "orientation": "square"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "url": photo["src"]["medium"],
                "photographer": photo.get("photographer", "Pexels"),
                "link": photo.get("url", "#"),
            }
            for photo in data.get("photos", [])
        ]
    except Exception:
        return []
