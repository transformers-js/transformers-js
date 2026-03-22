#!/usr/bin/env python3
"""Generate expected preprocessing outputs using Python transformers (or PIL fallback).

Outputs to benchmark/data/ (gitignored). Run once, then run compare.ts.

Usage:
    pip install -r benchmark/requirements.txt
    python benchmark/generate.py
"""
import json
import pathlib
import numpy as np
from PIL import Image

DATA = pathlib.Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

# Deterministic synthetic image — identical pattern in compare.ts.
# 256×256 so R=y, G=x, B=(y+x)//2 are exact integers [0,255]; no modulo wrap,
# no float rounding ambiguity, no bicubic-ringing discontinuities.
W, H = 256, 256
y_idx = np.arange(H, dtype=np.uint8).reshape(-1, 1) * np.ones(W, dtype=np.uint8)
x_idx = np.ones(H, dtype=np.uint8).reshape(-1, 1) * np.arange(W, dtype=np.uint8)
r = y_idx
g = x_idx
b = ((y_idx.astype(np.uint16) + x_idx.astype(np.uint16)) // 2).astype(np.uint8)
rgb = np.stack([r, g, b], axis=-1)
img = Image.fromarray(rgb, mode="RGB")

MODEL = "google/vit-base-patch16-224"

# Try Python transformers first; fall back to manual PIL if unavailable.
pixel_values = None
source = None
try:
    from transformers import AutoImageProcessor
    print(f"Using transformers AutoImageProcessor for {MODEL}")
    processor = AutoImageProcessor.from_pretrained(MODEL)
    inputs = processor(images=img, return_tensors="np")
    pixel_values = inputs["pixel_values"][0].astype(np.float32)
    source = "transformers"
except Exception as e:
    print(f"transformers unavailable ({e}), falling back to manual PIL preprocessing")

if pixel_values is None:
    # Manual ViT-base-patch16-224 config (resample=3=BICUBIC, a=-0.5 in PIL):
    #   size: {height:224, width:224}, rescale: 1/255
    #   image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5]
    arr = np.array(img.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32)
    arr = arr / 255.0
    arr = (arr - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])
    pixel_values = arr.transpose(2, 0, 1).astype(np.float32)  # CHW
    source = "manual-PIL"

out_path = DATA / "vit_pixel_values.bin"
pixel_values.tofile(out_path)

meta = {
    "model": MODEL,
    "source": source,
    "image_size": {"width": W, "height": H},
    "output_shape": list(pixel_values.shape),
    "output_min": float(pixel_values.min()),
    "output_max": float(pixel_values.max()),
    "output_mean": float(pixel_values.mean()),
}
(DATA / "vit_meta.json").write_text(json.dumps(meta, indent=2))

print(f"Saved {pixel_values.shape} tensor → {out_path}")
print(f"  min={meta['output_min']:.6f}  max={meta['output_max']:.6f}  mean={meta['output_mean']:.6f}")
