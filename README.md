# StyleSense — AI-Powered Fashion Recommendation System

StyleSense takes a photo of a garment and returns a real, contextual styling
recommendation — garment type, dominant color, an occasion & season-aware
outfit suggestion, and real reference photos for what to pair it with.

The project has two parts:

1. **`notebook/`** — a CNN trained from scratch on Fashion-MNIST in Google
   Colab, with full evaluation (accuracy/loss curves, confusion matrix,
   feature-map visualizations, misclassification analysis).
2. **The application** (everything else in this repo) — a full-stack local
   app built around a live vision + language pipeline.

## Why two different models?

Fashion-MNIST is 28×28 grayscale — excellent for demonstrating training and
evaluation fundamentals, but it has no color information and won't
generalize to real photos. Rather than force it onto live uploads, the CNN
is kept as the evaluated, from-scratch model (see `notebook/`), while the
live app uses **CLIP** — a zero-shot vision model that works directly on
real color photos with no additional training. This is a deliberate
engineering decision, not a limitation glossed over.

## How it works

| Step | Tool | What it does |
|---|---|---|
| 1 | **CLIP** (`openai/clip-vit-base-patch32`) | Zero-shot garment classification on the uploaded photo |
| 2 | **K-means clustering** (scikit-learn) | Extracts the dominant color from image pixels |
| 3 | **Groq API** (LLM) | Generates occasion & season-aware styling advice |
| 4 | **Pexels API** | Fetches real reference photos for the suggested pairing |
| 5 | **SQLite** | Saves every completed look, persisting across restarts |

**Frontend:** HTML, Tailwind CSS, vanilla JavaScript — a single-page app
with Upload, Results, Lookbook, Model Insights, and About views.

**Backend:** FastAPI (Python).

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get two free API keys

**Groq** (powers the styling advice text):
1. Go to [console.groq.com](https://console.groq.com), sign in
2. "API Keys" in the sidebar → "Create API Key" → copy it

**Pexels** (powers the pairing reference images):
1. Go to [pexels.com/api](https://www.pexels.com/api), sign up
2. Your API key is shown on your account page

### 3. Configure environment variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and paste in your real keys:
```
GROQ_API_KEY=your_actual_key_here
PEXELS_API_KEY=your_actual_key_here
```

### 4. Run it

```bash
uvicorn main:app --reload
```
(On Windows, if `uvicorn` isn't recognized directly, use `python -m uvicorn main:app --reload` instead.)

Open **http://localhost:8000** in your browser.

First request will take ~20-30 seconds while CLIP's model weights (~600MB)
download once — after that, every prediction is fast, using the local cache.

### Running without API keys

The app is intentionally built to **not break** without API keys — CLIP
detection, color extraction, the interface, and the Lookbook all work fully
offline. Without `GROQ_API_KEY`, styling text falls back to a templated
suggestion. Without `PEXELS_API_KEY`, pairing images are simply omitted with
a note. Both are clearly labeled in the UI when running in this fallback
mode.

## Project structure

```
.
├── notebook/
│   └── Fashion_Recommendation_System_almost_finishged.ipynb   # CNN training & evaluation (Colab)
├── main.py                          # FastAPI app & routes
├── db.py                            # SQLite persistence (Lookbook)
├── clip_classifier.py               # CLIP zero-shot garment detection
├── color_utils.py                   # K-means dominant color extraction
├── llm_groq.py                      # Groq LLM styling reasoning
├── pexels_images.py                 # Pexels reference image search
├── requirements.txt
├── .env.example                     # Template — copy to .env with real keys
└── static/
    ├── index.html                   # Frontend (single-page app)
    └── insights/                    # Exported evaluation plots from Colab
```

## Model evaluation summary

- **Dataset:** Fashion-MNIST — 70,000 grayscale images, 10 garment classes
- **Architecture:** Convolutional layers + max-pooling for feature
  extraction, dense layers with dropout for regularization, softmax output
- **Test accuracy:** 91.83%
- **Key finding:** the model's most common confusion was between shirts,
  coats, and pullovers — visually similar categories even at 28×28
  resolution. See `notebook/` and the app's Model Insights page for the full
  confusion matrix and feature-map visualizations.

## Built by

Diya Susan — BTech AI/ML, Christ University, Bangalore
