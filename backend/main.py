"""
main.py
-------
FastAPI backend for Route Resilience.

Two ways to see results, as requested:
  1. LIVE  — POST /predict with an uploaded satellite image tile.
             Runs your real trained model + graph analysis and returns
             the result image + stats.
  2. PRE-COMPUTED — GET /precomputed returns the saved results (the
             screenshots you already generated in Colab), so the
             dashboard has something to show even without a live model
             or GPU.

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import model_utils

app = FastAPI(title="Route Resilience API")

# Allow the frontend (opened as a local file or from localhost) to call this API
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


@app.on_event("startup")
def startup_event():
    """Try to load the model at startup. If it's missing, don't crash —
    the /precomputed endpoints still work; only /predict will 503."""
    try:
        model_utils.load_model()
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("⚠️  Live /predict will not work until the model file is added.")


@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": model_utils.is_model_loaded(),
        "endpoints": ["/predict (POST, image upload)", "/precomputed (GET)", "/health (GET)"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_utils.is_model_loaded()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Live inference: upload a satellite tile, get back the extraction +
    occlusion-cleanup + graph criticality result."""
    if not model_utils.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Add best_road_model.pt to backend/saved_model/ and restart the server.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        result = model_utils.run_full_pipeline(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return result


@app.get("/precomputed")
def get_precomputed():
    """Returns the manifest of saved Colab results, with image URLs
    pointing at the /static mount. Add your files to backend/precomputed/
    and list them in precomputed/manifest.json."""
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
