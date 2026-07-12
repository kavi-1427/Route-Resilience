"""
model_utils.py
--------------
Loads the trained U-Net road-extraction model and runs the full
extraction -> skeletonize -> graph -> criticality pipeline.

This is the same logic you built and validated in Colab, refactored
into a standalone module the FastAPI backend can import.
"""

import os
import io
import base64

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import networkx as nx
from skimage.morphology import skeletonize

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed on a server
import matplotlib.pyplot as plt


# ── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "best_road_model.pt")
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

_model = None  # loaded lazily, once, at server startup


def get_model():
    """Builds the U-Net architecture (must match what was trained)."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # weights come from the checkpoint, not ImageNet, at inference time
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(DEVICE)


def load_model():
    """Loads the trained checkpoint once. Call this at server startup."""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            f"Copy your best_road_model.pt from Google Drive into backend/saved_model/."
        )

    model = get_model()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _model = model
    print(f"Model loaded from {MODEL_PATH} | device={DEVICE} | best_iou={checkpoint.get('best_iou', 'unknown')}")
    return _model


def is_model_loaded() -> bool:
    return _model is not None


# ── STEP 1: road mask from raw image bytes ──────────────────────────
def predict_mask(image_bytes: bytes):
    """Runs the U-Net on an uploaded image and returns (rgb_image, binary_mask)."""
    model = load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    aug = _val_transform(image=image_np)
    inp = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(inp)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image_np, mask


# ── STEP 2: clean + skeletonize ──────────────────────────────────────
def clean_and_reconnect(mask, kernel_size=7, close_iterations=2, min_component_size=15):
    """Morphological cleanup: bridges small gaps, removes noise specks."""
    binary = (mask > 127).astype(np.uint8) * 255
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel_dilate, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=close_iterations)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clean = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_component_size:
            clean[labels == i] = 255
    return clean


def skeletonize_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    skeleton = skeletonize(mask_clean > 0).astype(np.uint8)
    return skeleton


# ── STEP 3: build the road graph ─────────────────────────────────────
def build_road_graph(skeleton):
    G = nx.Graph()
    road_pixels = np.argwhere(skeleton > 0)
    pixel_set = set(map(tuple, road_pixels))

    for (y, x) in road_pixels:
        G.add_node((y, x))

    for (y, x) in road_pixels:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in pixel_set:
                    dist = np.sqrt(dy ** 2 + dx ** 2)
                    G.add_edge((y, x), (ny, nx_), weight=dist)

    # simplify: collapse degree-2 nodes (points along a road, not intersections)
    nodes_to_remove = [n for n in G.nodes() if G.degree(n) == 2]
    for node in nodes_to_remove:
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 2:
            w1 = G[node][neighbors[0]].get("weight", 1)
            w2 = G[node][neighbors[1]].get("weight", 1)
            G.add_edge(neighbors[0], neighbors[1], weight=w1 + w2)
        G.remove_node(node)

    return G


# ── STEP 4: criticality scoring ──────────────────────────────────────
def calculate_criticality(G):
    if G.number_of_edges() == 0:
        return {}, {}, {}

    edge_centrality = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
    components_before = nx.number_connected_components(G)

    vulnerability = {}
    for edge in G.edges():
        G_temp = G.copy()
        G_temp.remove_edge(*edge)
        components_after = nx.number_connected_components(G_temp)
        vulnerability[edge] = 1 if components_after > components_before else 0

    max_cent = max(edge_centrality.values()) if edge_centrality else 1
    criticality = {}
    for edge in G.edges():
        cent = edge_centrality.get(edge, 0) / (max_cent + 1e-6)
        vuln = vulnerability.get(edge, 0)
        criticality[edge] = 0.7 * cent + 0.3 * vuln

    return criticality, edge_centrality, vulnerability


# ── STEP 5: visualization -> base64 PNG (for the API response) ──────
def render_result_image(image_np, mask, skeleton, criticality):
    """Builds the 4-panel result figure and returns it as a base64 PNG string."""
    criticality_map = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    colormap = plt.cm.RdYlGn_r
    for (node1, node2), score in criticality.items():
        y1, x1 = node1
        y2, x2 = node2
        color = colormap(score)
        rgb = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        cv2.line(criticality_map, (x1, y1), (x2, y2), rgb, 2)

    overlay = image_np.copy()
    crit_resized = cv2.resize(criticality_map, (image_np.shape[1], image_np.shape[0]))
    overlay = cv2.addWeighted(overlay, 0.45, crit_resized, 0.75, 0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    axes[0].imshow(image_np)
    axes[0].set_title("Satellite image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Extracted road mask")
    axes[1].axis("off")

    axes[2].imshow(skeleton, cmap="gray")
    axes[2].set_title("Skeleton (centerlines)")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Criticality: red = critical")
    axes[3].axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ── FULL PIPELINE — what the /predict endpoint calls ─────────────────
def run_full_pipeline(image_bytes: bytes):
    image_np, raw_mask = predict_mask(image_bytes)
    clean_mask = clean_and_reconnect(raw_mask)
    skeleton = skeletonize_mask(clean_mask)
    G = build_road_graph(skeleton)
    criticality, edge_centrality, vulnerability = calculate_criticality(G)
    result_image_b64 = render_result_image(image_np, clean_mask, skeleton, criticality)

    scores = list(criticality.values())
    top_edges = sorted(criticality.items(), key=lambda x: x[1], reverse=True)[:5]

    stats = {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "connected_components": int(nx.number_connected_components(G)),
        "avg_criticality": float(np.mean(scores)) if scores else 0.0,
        "max_criticality": float(np.max(scores)) if scores else 0.0,
        "road_pixel_percent": round(float(np.sum(clean_mask > 127)) / clean_mask.size * 100, 2),
        "top_segments": [
            {"rank": i + 1, "score": round(float(score), 4)}
            for i, (_, score) in enumerate(top_edges)
        ],
    }

    return {
        "result_image": result_image_b64,
        "stats": stats,
    }
