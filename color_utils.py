"""
Dominant color extraction from an uploaded garment photo.

Two things this handles carefully:
1. Which k-means cluster is actually "the garment" vs background/skin —
   picked by highest saturation among sufficiently-large clusters, since
   garments are usually more colorful than beige backgrounds or skin tones.
2. Naming that color accurately — hue-based matching (not raw RGB distance),
   since RGB-distance badly misclassifies dark saturated colors (e.g. navy
   blue landing "closer" to brown than to navy in pure euclidean terms).
"""

import colorsys
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "navy blue": (0, 0, 128),
    "blue": (0, 102, 204),
    "sky blue": (135, 206, 235),
    "red": (200, 30, 30),
    "maroon": (128, 0, 32),
    "pink": (255, 182, 193),
    "orange": (255, 140, 0),
    "yellow": (240, 220, 60),
    "mustard": (200, 160, 40),
    "green": (34, 139, 34),
    "olive": (110, 120, 40),
    "teal": (0, 128, 128),
    "purple": (128, 0, 128),
    "lavender": (200, 180, 230),
    "brown": (101, 67, 33),
    "tan": (210, 180, 140),
    "beige": (222, 202, 173),
    "cream": (245, 235, 210),
    "denim blue": (75, 100, 140),
}


def _closest_color_name(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

    if v < 0.22:
        return "black"
    if v > 0.92 and s < 0.12:
        return "white"
    if s < 0.14:
        return "grey"

    # Hue-based matching for everything else. Raw RGB euclidean distance
    # badly misjudges dark/saturated colors (a navy pixel can be
    # mathematically "closer" to brown than to navy blue) because it's
    # dominated by brightness differences, not perceived color. Matching
    # on hue first fixes that; saturation/value are only tie-breakers.
    chromatic = {
        name: rgb_val for name, rgb_val in NAMED_COLORS.items()
        if name not in ("black", "white", "grey")
    }
    best_name, best_score = "grey", float("inf")
    for name, (cr, cg, cb) in chromatic.items():
        ch, cs, cv = colorsys.rgb_to_hsv(cr / 255, cg / 255, cb / 255)
        hue_dist = min(abs(h - ch), 1 - abs(h - ch)) * 360  # circular, degrees
        score = hue_dist + 40 * abs(s - cs) + 25 * abs(v - cv)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def extract_dominant_color(image: Image.Image, k: int = 4) -> dict:
    """
    Returns {"name": "navy blue", "rgb": [r, g, b], "hex": "#1a2b3c"}
    """
    img = image.convert("RGB").resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    total = len(labels)

    # Candidate clusters: not near-white/near-black background noise,
    # and big enough to plausibly be the garment (not a stray highlight).
    candidates = []
    for idx, (r, g, b) in enumerate(centers):
        frac = counts[idx] / total
        is_near_white = r > 235 and g > 235 and b > 235
        is_near_black = r < 20 and g < 20 and b < 20
        if not is_near_white and not is_near_black and frac > 0.06:
            mx, mn = max(r, g, b), min(r, g, b)
            saturation = 0 if mx == 0 else (mx - mn) / mx
            candidates.append((saturation, frac, (r, g, b)))

    if candidates:
        # Prefer the most saturated (colorful) sizeable cluster — garments
        # are usually more colorful than beige backgrounds or skin tones.
        # Ties broken by cluster size.
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        chosen_rgb = candidates[0][2]
    else:
        # Nothing passed the filters (e.g. genuinely black/white garment) —
        # fall back to simple largest-cluster.
        order = np.argsort(-counts)
        chosen_rgb = tuple(centers[order[0]])

    r, g, b = [int(v) for v in chosen_rgb]
    name = _closest_color_name((r, g, b))
    hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return {"name": name, "rgb": [r, g, b], "hex": hex_code}
