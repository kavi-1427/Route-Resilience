# Route Resilience — VS Code Setup

This gives you **both** ways to see results, running together:

- **Live Demo** — upload any satellite tile, and your real trained U-Net model +
  graph criticality pipeline runs on it right there in the browser.
- **Saved Results** — the actual result images you already generated in Colab
  (fusion recovery, criticality maps, honest limitation tests), displayed in a
  gallery so the dashboard has content even without the model loaded.

## Folder structure

```
route-resilience/
├── backend/
│   ├── main.py              <- FastAPI server
│   ├── model_utils.py       <- U-Net + graph analysis (same logic as your Colab code)
│   ├── requirements.txt
│   ├── saved_model/         <- put best_road_model.pt here
│   └── precomputed/         <- your saved result images + manifest.json (already included)
└── frontend/
    └── index.html           <- the dashboard (just open it in a browser)
```

## Step 1 — Open the folder in VS Code

`File → Open Folder…` → select `route-resilience`.

## Step 2 — Add your trained model

Copy `best_road_model.pt` (the one you downloaded from Google Drive at the end
of training) into:

```
backend/saved_model/best_road_model.pt
```

If you skip this step, the **Live Demo** tab will show a clear error message,
but **Saved Results** still works fine — that's the point of having both.

## Step 3 — Set up the Python environment

Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

This installs PyTorch, FastAPI, and the same libraries you used in Colab
(segmentation-models-pytorch, networkx, scikit-image, etc).

**Note:** installing PyTorch can take a few minutes and several GB of disk
space, especially if pip pulls a CUDA build you don't need. If you don't have
a local GPU, that's fine — inference will just run on CPU (a few seconds per
image, not a problem for a single-image demo).

## Step 4 — Run the backend

Still inside `backend/`, with the virtual environment active:

```bash
uvicorn main:app --reload --port 8000
```

You should see:

```
✅ Model loaded from .../saved_model/best_road_model.pt | device=cpu | best_iou=0.5852
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Leave this terminal running.

## Step 5 — Open the frontend

Just open `frontend/index.html` directly in your browser (double-click it, or
right-click → "Open with Live Server" if you have that VS Code extension).

The top-right badge will show **"model loaded — live demo ready"** once it can
reach the backend.

## What each tab does

- **Live Demo** — drag and drop any satellite image tile. It calls
  `POST /predict` on your backend, which runs your actual U-Net, cleans up the
  mask, builds the graph, scores criticality, and returns a 4-panel result
  image plus live stats (nodes, edges, connected components, max criticality).

- **Saved Results** — calls `GET /precomputed`, which reads
  `backend/precomputed/manifest.json` and serves the images already sitting in
  that folder. This is exactly the fusion/criticality/limitation screenshots
  you generated in Colab — already wired up and included in this package.

- **About** — a plain-language writeup of the full pipeline and its known
  limitation, pulled from what you actually tested (no invented steps).

## Adding more precomputed results later

Drop any new image into `backend/precomputed/`, then add an entry to
`backend/precomputed/manifest.json`:

```json
{
  "id": "unique_id",
  "filename": "your_image.jpg",
  "title": "Short title",
  "description": "One or two sentences.",
  "category": "fusion",
  "stats": {}
}
```

`category` can be `fusion`, `criticality`, or `limitations` — these power the
filter buttons in the gallery.

## Deploying for submission

The frontend is a single static HTML file — deploy it as-is to GitHub Pages
or Netlify. The backend needs an actual Python host (Render, Railway, or a
free-tier VM) since it runs your model — or, for the hackathon demo itself,
it's completely fine to just run `uvicorn` locally on your laptop and present
from there. Judges seeing a live upload work in person is often more
convincing than a slow free-tier server anyway.
