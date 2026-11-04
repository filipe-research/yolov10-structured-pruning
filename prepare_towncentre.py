#!/usr/bin/env python3
"""
Converte o Oxford Town Centre (video + ground truth .top) para o formato YOLO
esperado pelo pipeline (images/{train,val} + labels/{train,val} + data.yaml).

Formato do TownCentre-groundtruth.top (CSV, uma deteccao por linha):
  personNumber, frameNumber, headValid, bodyValid,
  headLeft, headTop, headRight, headBottom,
  bodyLeft, bodyTop, bodyRight, bodyBottom

Usamos a caixa de CORPO (bodyValid == 1), classe unica "pessoa".

Split TEMPORAL por padrao (primeiros --train-frac dos frames -> train, resto
-> val): frames consecutivos de video sao quase identicos, entao split
aleatorio por frame vaza informacao do treino para a validacao e infla o mAP.
ATENCAO: para comparar numeros com uma tabela ja existente, use o MESMO split
dessa tabela (o split temporal daqui nao e necessariamente o mesmo).

Uso:
  python prepare_towncentre.py --video TownCentreXVID.mp4 \
      --gt TownCentre-groundtruth.top --out oxford [--train-frac 0.7] [--stride 1]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--gt", required=True, help="TownCentre-groundtruth.top")
    p.add_argument("--out", default="oxford")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--stride", type=int, default=1,
                   help="Usar 1 frame a cada N (reduz tamanho/correlacao)")
    return p.parse_args()


def load_groundtruth(gt_path):
    boxes = defaultdict(list)   # frame -> [(x1,y1,x2,y2)]
    with open(gt_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 12:
                continue
            try:
                frame = int(float(row[1]))
                body_valid = int(float(row[3]))
                x1, y1, x2, y2 = (float(v) for v in row[8:12])
            except ValueError:
                continue
            if body_valid == 1:
                boxes[frame].append((x1, y1, x2, y2))
    return boxes


def main():
    args = parse_args()
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    gt = load_groundtruth(args.gt)
    frames_gt = sorted(gt)
    split_frame = frames_gt[int(len(frames_gt) * args.train_frac)]
    print(f"{len(frames_gt)} frames anotados | split temporal no frame {split_frame} "
          f"(train < {split_frame} <= val)")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Nao foi possivel abrir o video: {args.video}")

    n_saved = {"train": 0, "val": 0}
    idx = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        idx += 1
        if idx not in gt or (idx % args.stride) != 0:
            continue
        h, w = img.shape[:2]
        split = "train" if idx < split_frame else "val"
        name = f"frame_{idx:06d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for x1, y1, x2, y2 in gt[idx]:
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
        n_saved[split] += 1

    cap.release()
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: pessoa\n")
    print(f"OK: train={n_saved['train']} imgs, val={n_saved['val']} imgs")
    print(f"data.yaml: {(out / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
