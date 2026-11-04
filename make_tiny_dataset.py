#!/usr/bin/env python3
"""Gera um mini-dataset sintetico (8 imgs, 1 classe) so para smoke test do
pipeline (poda -> fine-tuning -> avaliacao) sem depender da base real.
Uso:  python make_tiny_dataset.py  ->  tiny_dataset/data.yaml"""
import numpy as np
import cv2
from pathlib import Path

rng = np.random.default_rng(0)
root = Path("tiny_dataset")
for split in ("train", "val"):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(8):
        img = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
        x, y, w, h = rng.integers(40, 120, 4)
        img[y:y + h, x:x + w] = 220
        cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
        xc, yc = (x + w / 2) / 320, (y + h / 2) / 320
        (root / "labels" / split / f"{i}.txt").write_text(
            f"0 {xc:.4f} {yc:.4f} {w / 320:.4f} {h / 320:.4f}\n")
(root / "data.yaml").write_text(
    f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
    "names:\n  0: pessoa\n")
print("dataset ok:", root.resolve())
