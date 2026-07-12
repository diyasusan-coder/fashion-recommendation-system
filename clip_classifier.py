"""
Zero-shot garment classification using OpenAI's CLIP (via HuggingFace transformers).

Unlike the Fashion-MNIST CNN (28x28 grayscale, 10 generic classes), CLIP works
directly on full-color real photos and can be pointed at a much richer, more
"presentation impressive" label set without any extra training. Model downloads
once (~600MB) the first time you run the app, then it's cached locally.
"""

from functools import lru_cache
from PIL import Image
import torch

# Rich candidate labels — feel free to trim/expand this list.
GARMENT_LABELS = [
    "a t-shirt", "a formal shirt", "a casual shirt", "a hoodie", "a sweater",
    "a denim jacket", "a leather jacket", "a blazer", "a coat",
    "a pair of jeans", "a pair of trousers", "a pair of shorts",
    "a casual dress", "a formal gown", "a floral dress", "a kurta",
    "a saree", "a skirt", "a pair of sneakers", "a pair of formal shoes",
    "a pair of sandals", "a handbag", "a backpack",
]


@lru_cache(maxsize=1)
def _load_clip():
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def classify_garment(image: Image.Image, top_k: int = 3):
    """
    Returns a list of {"label": str, "confidence": float}, sorted descending,
    using CLIP zero-shot classification against GARMENT_LABELS.

    Raises on failure (e.g. no internet on first run to download weights) —
    caller is expected to catch and fall back gracefully.
    """
    model, processor = _load_clip()

    inputs = processor(
        text=GARMENT_LABELS,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    scored = sorted(
        zip(GARMENT_LABELS, probs.tolist()), key=lambda x: -x[1]
    )[:top_k]

    return [
        {"label": label.replace("a pair of ", "").replace("a ", ""), "confidence": round(conf, 3)}
        for label, conf in scored
    ]
