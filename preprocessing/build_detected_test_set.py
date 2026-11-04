#!/usr/bin/env python3
"""A test set of detected objects: LangSAM finds each test image's class object,
and the test sample is
that object's box (+ context), cut in memory. Only JSON is written
(helpers.image_refs crop references):

    test.json        {"context", "crops": [{"image", "bbox": [x, y, w, h],
    "label", "score"}]}  (IMAGE_ROOT_SINGLE)
    grids.json       {"cell_size", "grids": [{"cells": [crop refs], "labels"}]}
    (IMAGE_ROOT_GRID)
    detections.json  every image: the class's detections (boxes, scores) or
    none, for auditing

A full COCO image shows several of its classes while it is scored against one
(its folder); a box around
the detected class object is one object, as in ImageNet. Per image the
highest-scoring box of its class is
kept if it scores >= --box-threshold and its sides are >= --min-side px; other
images are left out.

    python preprocessing/build_detected_test_set.py \\
        --test-dir data/coco_common10/val \\
        --out data/coco_common10/detected_test --device cuda:0
"""

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from tgcl.helpers.image_refs import crop_ref
from tgcl.helpers.image_refs import IMAGE_EXTENSIONS
from tgcl.langsam_utils import load_langsam

ROOT = Path(__file__).resolve().parent.parent


def best_detection(result: dict, min_side: float) -> Optional[Dict]:
    """The highest-scoring box (xyxy -> xywh) of one LangSAM result whose sides
    are >= min_side."""
    boxes = np.asarray(result.get("boxes", []), dtype=float).reshape(-1, 4)
    scores = np.asarray(result.get("scores", []), dtype=float).reshape(-1)
    best = None
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        w, h = x2 - x1, y2 - y1
        if min(w, h) >= min_side and (best is None or s > best["score"]):  # pylint: disable=unsubscriptable-object
            best = {
                "bbox": [
                    round(float(x1), 1),
                    round(float(y1), 1),
                    round(float(w), 1),
                    round(float(h), 1),
                ],
                "score": round(float(s), 4),
            }
    return best


def all_detections(result: dict) -> List[Dict]:
    boxes = np.asarray(result.get("boxes", []), dtype=float).reshape(-1, 4)
    scores = np.asarray(result.get("scores", []), dtype=float).reshape(-1)
    return [
        {"xyxy": [round(float(v), 1) for v in b], "score": round(float(s), 4)}
        for b, s in zip(boxes, scores)
    ]


def build_grids(
    crops: List[Dict], spec_path: Path, num_grids: int, grid_n: int, seed: int
) -> List[Dict]:
    """2x2 (grid_n) grids of crops of distinct classes, one random crop per
    cell."""
    by_class: Dict[str, List[int]] = {}
    for i, c in enumerate(crops):
        by_class.setdefault(c["label"], []).append(i)
    classes = sorted(by_class)
    if len(classes) < grid_n:
        raise SystemExit(
            f"{grid_n}-cell grids need {grid_n} classes with detections, "
            f"got {len(classes)}"
        )
    rng = random.Random(seed)
    grids = []
    for _ in range(num_grids):
        labels = rng.sample(classes, grid_n)
        grids.append(
            {
                "cells": [
                    crop_ref(spec_path, rng.choice(by_class[c])) for c in labels
                ],
                "labels": labels,
            }
        )
    return grids


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="<class folder>/<image> test images",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument(
        "--min-side",
        type=float,
        default=16.0,
        help="smallest box side kept (image px)",
    )
    ap.add_argument(
        "--context",
        type=float,
        default=0.1,
        help="margin per side, fraction of the box's longer side",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-grids", type=int, default=50)
    ap.add_argument("--grid-n", type=int, default=4)
    ap.add_argument("--cell-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = load_langsam(device=args.device)

    crops, audit = [], {}
    for folder in sorted(p for p in args.test_dir.iterdir() if p.is_dir()):
        label = folder.name.replace("_", " ")
        files = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        kept = 0
        for start in range(0, len(files), args.batch_size):
            batch = files[start : start + args.batch_size]
            images = [Image.open(f).convert("RGB") for f in batch]
            results = model.predict(
                images,
                [label] * len(images),
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            for f, res in zip(batch, results):
                rel = (
                    str(f.relative_to(ROOT))
                    if f.is_absolute() and ROOT in f.parents
                    else str(f)
                )
                audit[rel] = {"label": label, "detections": all_detections(res)}
                best = best_detection(res, args.min_side)
                if best is not None:
                    crops.append({"image": rel, "label": label, **best})
                    kept += 1
        print(
            f"  {label:13s} {kept}/{len(files)} images with a detection",
            flush=True,
        )
        if kept == 0:
            raise SystemExit(
                f"no detection for {label!r}: lower --box-threshold or "
                "check the test images"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    spec = args.out / "test.json"
    json.dump(
        {"context": args.context, "crops": crops},
        open(spec, "w", encoding="utf-8"),
        indent=1,
    )
    json.dump(
        {
            "cell_size": args.cell_size,
            "grids": build_grids(
                crops, spec, args.num_grids, args.grid_n, args.seed
            ),
        },
        open(args.out / "grids.json", "w", encoding="utf-8"),
        indent=1,
    )
    json.dump(
        {
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "images": audit,
        },
        open(args.out / "detections.json", "w", encoding="utf-8"),
        indent=1,
    )
    print(
        f"{len(crops)} detected objects -> {spec} (+ grids.json, "
        "detections.json; JSON only)"
    )


if __name__ == "__main__":
    main()
