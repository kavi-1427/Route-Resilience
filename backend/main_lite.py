"""
main_lite.py
------------
Lightweight FastAPI backend for Route Resilience — hosted version.

This version does NOT load PyTorch/OpenCV (too heavy for free-tier hosting).
It serves only the precomputed gallery results which is enough to demo the project.

Run with:
    uvicorn main_lite:app --reload --port 8000
"""

import os
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Route Resilience API (Lite)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PRECOMPUTED_DIR = os.path.join(os.path.dirname(__file__), "precomputed")
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

# Serve the precomputed images as static files at /static/...
app.mount("/static", StaticFiles(directory=PRECOMPUTED_DIR), name="static")


@app.get("/")
def root():
    return {
        "status": "ok",
        "mode": "lite — precomputed results only",
        "note": "Live /predict is disabled in the hosted free-tier version.",
        "endpoints": ["/precomputed (GET)", "/health (GET)"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": False,
        "mode": "lite",
    }


@app.post("/predict")
def predict_disabled():
    return {
        "error": "Live inference is not available on the free-tier hosted version. "
                 "Run the backend locally with the full requirements to use live prediction."
    }


@app.get("/precomputed")
def get_precomputed():
    """Returns the manifest of saved results with image URLs."""
    manifest_path = os.path.join(PRECOMPUTED_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return {"items": []}

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    for item in manifest.get("items", []):
        item["image_url"] = f"/static/{item['filename']}"

    return manifest


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_lite:app", host="0.0.0.0", port=8000, reload=True)
