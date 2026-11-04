#!/usr/bin/env python3
"""Draw TGCL explanations: test image, generated answer, and per answer token
its top concepts.

One row per explained image. For each answer token: the top-k concepts with
their
cosine similarity to the token's hidden state, the concept's class name, its
words
(text grounding, decoded from the concept direction) and its grounding crops
(the
training segments with the highest concept activation).

    RUN=outputs/tgcl_sam_coco_seed1
    python scripts/plot_explanations.py \\
        --explanations $RUN/explanations/snmf/vlm_explanations.json \\
        --classes bench,truck,cup --out assets/tgcl_example_coco.png
"""

import argparse
from pathlib import Path
import sys
from typing import List

import matplotlib

matplotlib.use("Agg")  # a file backend, before pyplot is imported
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
from PIL import Image

from tgcl.helpers.explanations import load
from tgcl.helpers.grounding import grounding_crop

# pylint: enable=wrong-import-position

ROOT = Path(__file__).resolve().parent.parent


def _path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def pick(results: List[dict], classes: List[str]) -> List[dict]:
    """First correctly answered image of each class whose rank-1 concept is that
    class."""
    chosen = []
    for cls in classes:
        for r in results:
            toks = [
                t
                for t in r.get("per_token_concepts") or []
                if t["token_text"].strip()
            ]
            if Path(r["image_path"]).parent.name != cls or not toks:
                continue
            top = toks[0]["top_concepts"][0]
            if (
                cls in r["model_output"].lower()
                and (top.get("concept_name") or [None])[0] == cls
            ):
                chosen.append(r)
                break
    return chosen


def plot(
    records: List[dict],
    out: Path,
    top_k: int,
    n_crops: int,
    n_words: int,
    image_width: int,
) -> None:
    """Plot the explanations of the given records into one figure.

    Per record: the test image and the answer, then per answer token its top_k
    concepts with their words (n_words) and grounding crops (n_crops).
    """
    rows = sum(
        min(top_k, len(t["top_concepts"]))
        for r in records
        for t in r["per_token_concepts"]
        if t["token_text"].strip()
    )
    fig = plt.figure(
        figsize=(2.2 * (2 + n_crops), 1.25 * rows + 0.6 * len(records))
    )
    grid = fig.add_gridspec(
        rows,
        2 + n_crops,
        width_ratios=[2.2, 2.4] + [1] * n_crops,
        hspace=0.35,
        wspace=0.08,
    )
    row = 0
    for r in records:
        toks = [t for t in r["per_token_concepts"] if t["token_text"].strip()]
        span = sum(min(top_k, len(t["top_concepts"])) for t in toks)
        ax = fig.add_subplot(grid[row : row + span, 0])
        ax.imshow(Image.open(_path(r["image_path"])).convert("RGB"))
        ax.set_title(f"answer: \"{r['model_output'].strip()}\"", fontsize=10)
        ax.axis("off")
        for t in toks:
            for c in t["top_concepts"][:top_k]:
                ax = fig.add_subplot(grid[row, 1])
                ax.axis("off")
                words = ", ".join(c.get("text_grounding", [])[:n_words])
                ax.text(
                    0,
                    0.5,
                    f"token \"{t['token_text'].strip()}\" → rank {c['rank']}\n"
                    f"concept: {c['concept_name'][0]}  (cos "
                    f"{c['similarity']:.2f})\nwords: {words}",
                    fontsize=8.5,
                    va="center",
                    ha="left",
                    transform=ax.transAxes,
                )
                paths = c.get("image_grounding_path", [])[:n_crops]
                boxes = c.get("image_grounding_bboxes") or [None] * len(paths)
                for j, (p, b) in enumerate(zip(paths, boxes)):
                    ax = fig.add_subplot(grid[row, 2 + j])
                    ax.imshow(grounding_crop(_path(p), b, image_width))
                    ax.axis("off")
                row += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--explanations",
        type=Path,
        required=True,
        help="vlm_explanations.json of step 6",
    )
    ap.add_argument(
        "--classes",
        default="",
        help="comma-separated: one correctly explained image per class",
    )
    ap.add_argument(
        "--indices",
        default="",
        help="comma-separated result indices (instead of --classes)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--words", type=int, default=5)
    ap.add_argument(
        "--image-width",
        type=int,
        default=512,
        help="IMAGE_SIZE_WIDTH of the run (grounding boxes are in images "
        "resized to this width)",
    )
    args = ap.parse_args()
    results = load(args.explanations)["results"]
    if args.indices:
        records = [results[int(i)] for i in args.indices.split(",")]
    else:
        records = pick(results, [c for c in args.classes.split(",") if c])
    if not records:
        sys.exit("no matching explanations")
    plot(
        records, args.out, args.top_k, args.crops, args.words, args.image_width
    )


if __name__ == "__main__":
    main()
