"""
Fashion Recommendation System — Full-Stack Demo
=================================================
Run with:  uvicorn main:app --reload
Then open: http://localhost:8000
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

import db
import color_utils
import clip_classifier
import llm_groq
import pexels_images

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Fashion Recommendation System")

db.init_db()


@app.on_event("startup")
def warm_up_clip():
    # Load CLIP once at startup so the first user request isn't slow.
    # If it fails (e.g. no internet yet on very first run), we just log it —
    # classify_garment() will raise per-request and we fall back gracefully.
    try:
        clip_classifier._load_clip()
        print("[startup] CLIP model loaded and ready.")
    except Exception as e:
        print(f"[startup] CLIP not preloaded yet ({e}). Will retry on first request.")


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    occasion: str = Form(...),
    season: str = Form(...),
):
    # Save the uploaded image
    ext = Path(image.filename).suffix or ".jpg"
    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = UPLOAD_DIR / saved_name
    contents = await image.read()
    with open(saved_path, "wb") as f:
        f.write(contents)

    pil_image = Image.open(saved_path)

    # 1. Garment classification (CLIP zero-shot, falls back to "clothing item")
    try:
        predictions = clip_classifier.classify_garment(pil_image, top_k=3)
        garment = predictions[0]["label"]
    except Exception as e:
        print(f"[predict] CLIP classification failed: {e}")
        predictions = []
        garment = "clothing item"

    # 2. Dominant color extraction
    try:
        color = color_utils.extract_dominant_color(pil_image)
    except Exception as e:
        print(f"[predict] Color extraction failed: {e}")
        color = {"name": "neutral", "rgb": [128, 128, 128], "hex": "#808080"}

    # 3. LLM outfit reasoning (Groq, falls back to templated suggestion)
    llm_result = llm_groq.get_outfit_recommendation(garment, color["name"], occasion, season)

    # 4. Real pairing images (Pexels, falls back to empty list)
    pairing_images = pexels_images.search_pairing_images(llm_result["pairing_query"])

    # 5. Persist to Lookbook history
    entry_id = db.save_entry(
        image_path=f"/static/uploads/{saved_name}",
        garment=garment,
        color_name=color["name"],
        color_hex=color["hex"],
        occasion=occasion,
        season=season,
        suggestion=llm_result["suggestion"],
        pairing_images=pairing_images,
    )

    return JSONResponse({
        "id": entry_id,
        "image_path": f"/static/uploads/{saved_name}",
        "garment": garment,
        "top_predictions": predictions,
        "color": color,
        "occasion": occasion,
        "season": season,
        "suggestion": llm_result["suggestion"],
        "suggestion_source": llm_result["source"],
        "pairing_images": pairing_images,
    })


@app.get("/api/history")
def history(limit: int = 50):
    return db.get_history(limit=limit)


@app.delete("/api/history/{entry_id}")
def delete_history(entry_id: int):
    db.delete_entry(entry_id)
    return {"status": "deleted", "id": entry_id}


# --- Serve the frontend ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "static" / "index.html")
