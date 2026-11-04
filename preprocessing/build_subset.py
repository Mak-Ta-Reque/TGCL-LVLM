#!/usr/bin/env python3
"""A class subset of a built dataset, as JSON maps only: no image is copied,
linked or re-encoded.

From a dataset laid out like data/imagenet1k (concepts_to_images.json: class ->
training image file
names in train_all/; val/<class folder>/ test images), writes to --out:

    concepts_to_images.json   the subset's classes -> their training images
    (CONCEPT_MAP, step 1)
    vocab.txt                 the subset's class names (CONCEPTS_VOCAB,
    EXPL_CHOICES_FILE)
    test.json                 {"images": [val paths]}
    (IMAGE_ROOT_SINGLE)
    grids.json                {"cell_size", "grids": [{"cells", "labels"}]}
    (IMAGE_ROOT_GRID)

Grids hold n distinct classes each, one random test image per cell, and are
composed in memory when
evaluated (helpers.image_refs). Example:

    python preprocessing/build_subset.py --source data/imagenet1k \\
        --out data/imagenet1k/subsets/imagenet5 \\
        --classes "goldfish,toucan,school bus,coffee mug,acorn"
"""

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List


def folder_of(source: Path, cls: str) -> Path:
    for name in (cls, cls.replace(" ", "_")):
        if (source / "val" / name).is_dir():
            return source / "val" / name
    raise FileNotFoundError(
        f"no test folder for class {cls!r} under {source / 'val'}"
    )


def build(
    source: Path,
    classes: List[str],
    num_grids: int,
    grid_n: int,
    cell_size: int,
    seed: int,
    grids_only: bool = False,
    train_per_class: int = 0,
    test_per_class: int = 0,
) -> Dict:
    """grids_only: only test.json and grids.json (the source needs just
    val/<class>/). train_per_class / test_per_class > 0: the first that many
    images of each class (quick runs).
    """
    if not classes:
        classes = sorted(
            p.name.replace("_", " ")
            for p in (source / "val").iterdir()
            if p.is_dir()
        )
    concept_map = (
        {}
        if grids_only
        else json.load(
            open(source / "concepts_to_images.json", encoding="utf-8")
        )
    )
    missing = [c for c in classes if not grids_only and c not in concept_map]
    if missing:
        raise SystemExit(
            f"classes not in {source / 'concepts_to_images.json'}: {missing}"
        )
    if len(classes) < grid_n:
        raise SystemExit(
            f"{grid_n}-cell grids need at least {grid_n} classes, got "
            f"{len(classes)}"
        )
    tests = {
        c: sorted(str(p) for p in folder_of(source, c).iterdir() if p.is_file())
        for c in classes
    }
    if test_per_class > 0:
        tests = {c: files[:test_per_class] for c, files in tests.items()}
    if train_per_class > 0:
        concept_map = {
            c: list(concept_map[c])[:train_per_class] for c in classes
        }
    rng = random.Random(seed)
    grids = []
    for _ in range(num_grids):
        labels = rng.sample(classes, grid_n)
        grids.append(
            {"cells": [rng.choice(tests[c]) for c in labels], "labels": labels}
        )
    files = {
        "test.json": {"images": [p for c in classes for p in tests[c]]},
        "grids.json": {"cell_size": cell_size, "grids": grids},
    }
    if not grids_only:
        files["concepts_to_images.json"] = {c: concept_map[c] for c in classes}
        files["vocab.txt"] = "\n".join(classes) + "\n"
    return files


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--classes",
        default="",
        help="comma-separated class names as in concepts_to_images.json "
        "(default: every val/ folder)",
    )
    ap.add_argument(
        "--grids-only",
        action="store_true",
        help="write only test.json and grids.json (all classes of the source's "
        "val/)",
    )
    ap.add_argument(
        "--train-per-class",
        type=int,
        default=0,
        help="first N training images per class (0 = all)",
    )
    ap.add_argument(
        "--test-per-class",
        type=int,
        default=0,
        help="first N test images per class (0 = all)",
    )
    ap.add_argument("--num-grids", type=int, default=50)
    ap.add_argument(
        "--grid-n", type=int, default=4, help="cells per grid (a square number)"
    )
    ap.add_argument("--cell-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = build(
        args.source,
        [c.strip() for c in args.classes.split(",") if c.strip()],
        args.num_grids,
        args.grid_n,
        args.cell_size,
        args.seed,
        grids_only=args.grids_only,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        with open(args.out / name, "w", encoding="utf-8") as f:
            f.write(
                content
                if isinstance(content, str)
                else json.dumps(content, indent=1)
            )
    n_train = sum(
        len(v) for v in files.get("concepts_to_images.json", {}).values()
    )
    print(
        f"{args.out}: {len(files['test.json']['images'])} test images, "
        f"{len(files['grids.json']['grids'])} grids, "
        f"{n_train} training images (JSON only)"
    )


if __name__ == "__main__":
    main()
