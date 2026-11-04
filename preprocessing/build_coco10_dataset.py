#!/usr/bin/env python3
"""
Build a COCO category subset (default: "coco10") analogous to data/imagenet10.

Source: real COCO train2017 (train split) + val2017 (test/val split) under
--coco-root (default $COCO_ROOT, else data/coco_raw):
    <coco-root>/annotations/{instances,captions}_{train,val}2017.json
    <coco-root>/{train2017,val2017}/<file>.jpg
train2017 and val2017 are disjoint official pools, so no re-split is needed.

With --download, missing files are fetched from images.cocodataset.org: the
annotation archive once, then ONLY the images selected for the subset (each
verified, written atomically) -- not the full 19 GB image sets.

Categories (COCO names, e.g. "dining table", "donut"):
    default             apple, banana, bird, cake, cat, cup, dog, donut, knife,
                        orange
    --categories A,B    an explicit list
    --top-k-common K    the K categories present in the most train2017 images

Per category, up to --train-cap (default 300) images go to train (random,
seeded) and up to --test-cap (default 50) to test (largest annotated area
first, so the object stays legible when cropped).

Output layout (mirrors data/imagenet10):
    <output-dir>/train_all/<file>.jpg       flat pool of all categories' train
                                            images
    <output-dir>/val/<category>/<file>.jpg  per-category test images
    <output-dir>/manifest.json              categories, per-category train/test
                                            image ids, file names, annotations

Usage:
    python preprocessing/build_coco10_dataset.py           # coco10
    python preprocessing/build_coco10_dataset.py --top-k-common 10 \\
        --download --output-dir data/coco_common10         # paper ablation set
    python preprocessing/build_coco10_dataset.py --train-cap 20 \\
        --test-cap 5                                       # pilot subset
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Optional, Tuple
import urllib.request
import zipfile

from PIL import Image

ROOT_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "coco10"
DEFAULT_COCO_ROOT = Path(
    os.environ.get("COCO_ROOT", ROOT_DIR / "data" / "coco_raw")
)

DEFAULT_CATEGORIES = [
    "apple",
    "banana",
    "bird",
    "cake",
    "cat",
    "cup",
    "dog",
    "donut",
    "knife",
    "orange",
]

COCO_SERVER = "http://images.cocodataset.org"
ANNOTATIONS_ARCHIVE = "annotations/annotations_trainval2017.zip"
SPLITS = {"train": "train2017", "test": "val2017"}


# ---------------------------------------------------------------- COCO files


def annotations_path(coco_root: Path, coco_split: str) -> Path:
    return coco_root / "annotations" / f"instances_{coco_split}.json"


def _download(
    url: str, dest: Path, retries: int = 3, timeout: int = 60
) -> None:
    """Download url to dest atomically (temp file + rename), with retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with (
                urllib.request.urlopen(url, timeout=timeout) as response,
                open(tmp, "wb") as out,
            ):
                shutil.copyfileobj(response, out)
            os.replace(tmp, dest)
            return
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {url}"
                ) from exc


def ensure_annotations(coco_root: Path, download: bool) -> None:
    """Make sure instances_ and captions_{train,val}2017.json exist (fetching
    them if allowed).

    The captions feed scripts/build_coco_captions.py (concept-word alignment
    BERTScore).
    """
    needed = [annotations_path(coco_root, s) for s in SPLITS.values()]
    needed += [
        coco_root / "annotations" / f"captions_{s}.json"
        for s in SPLITS.values()
    ]
    if all(p.exists() for p in needed):
        return
    if not download:
        missing = [str(p) for p in needed if not p.exists()]
        raise FileNotFoundError(
            f"Missing COCO annotations {missing}; pass --download or set "
            "--coco-root."
        )
    archive = coco_root / "annotations_trainval2017.zip"
    if not archive.exists():
        print(
            f"Downloading {COCO_SERVER}/{ANNOTATIONS_ARCHIVE} (~241 MB) ...",
            flush=True,
        )
        _download(f"{COCO_SERVER}/{ANNOTATIONS_ARCHIVE}", archive)
    with zipfile.ZipFile(archive) as zf:
        for path in needed:
            zf.extract(f"annotations/{path.name}", coco_root)
    archive.unlink()


def _verify_image(path: Path) -> None:
    with Image.open(path) as img:
        img.verify()


def ensure_images(
    coco_root: Path,
    coco_split: str,
    file_names: List[str],
    download: bool,
    workers: int = 16,
) -> None:
    """Make sure the listed images of one split exist (fetching only those if
    allowed)."""
    image_dir = coco_root / coco_split
    missing = sorted({f for f in file_names if not (image_dir / f).exists()})
    if not missing:
        return
    if not download:
        raise FileNotFoundError(
            f"{len(missing)} {coco_split} images missing under "
            f"{image_dir}; pass --download."
        )
    print(f"Downloading {len(missing)} {coco_split} images ...", flush=True)

    def fetch(file_name: str) -> None:
        dest = image_dir / file_name
        _download(f"{COCO_SERVER}/{coco_split}/{file_name}", dest)
        try:
            _verify_image(dest)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Corrupt download: {dest}") from exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, missing))


def load_annotations(path: Path):
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------- selection


def rank_categories_by_images(coco_json: dict) -> List[Tuple[str, int]]:
    """[(category, number of images containing it)], most frequent first."""
    cat_id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    images_per_cat: Dict[str, set] = defaultdict(set)
    for ann in coco_json["annotations"]:
        images_per_cat[cat_id_to_name[ann["category_id"]]].add(ann["image_id"])
    return sorted(
        ((name, len(ids)) for name, ids in images_per_cat.items()),
        key=lambda x: (-x[1], x[0]),
    )


def build_indices(coco_json: dict, categories: List[str]):
    """Index a COCO annotation file by the chosen categories.

    Returns:
        (category name -> id, image id -> file name,
        image id -> {category name: [annotations]}).

    Raises:
        ValueError: A category is not in the file.
    """
    cat_id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    target_name_to_id = {
        name: cid for cid, name in cat_id_to_name.items() if name in categories
    }
    missing = set(categories) - set(target_name_to_id.keys())
    if missing:
        raise ValueError(
            f"Category names not found in COCO categories: {missing}"
        )

    image_id_to_filename = {
        img["id"]: img["file_name"] for img in coco_json["images"]
    }

    # image_id -> {category_name: [annotation, ...]}
    img_cat_annotations: Dict[int, Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ann in coco_json["annotations"]:
        cid = ann["category_id"]
        if cid in cat_id_to_name and cat_id_to_name[cid] in categories:
            img_cat_annotations[ann["image_id"]][cat_id_to_name[cid]].append(
                ann
            )

    return target_name_to_id, image_id_to_filename, img_cat_annotations


def select_per_category(
    train_img_cat_annotations: Dict[int, Dict[str, List[dict]]],
    test_img_cat_annotations: Dict[int, Dict[str, List[dict]]],
    categories: List[str],
    seed: int,
    train_cap: int,
    test_cap: int,
):
    """Pick each category's train and test images.

    Train images are a seeded random sample; test images are those with the
    largest annotated area of the category (small instances make unreadable
    crops).

    Returns:
        {category: {"train_image_ids": [...], "test_image_ids": [...]}}.
    """
    rng = random.Random(seed)
    per_category = {}
    for category in categories:
        train_candidates = sorted(
            img_id
            for img_id, cats in train_img_cat_annotations.items()
            if category in cats
        )
        rng.shuffle(train_candidates)

        # Validation crops are ground-truth-mask crops shown directly to the
        # VLM -- many COCO instances are tiny, which upscales to blurry,
        # unrecognizable crops. Prefer the largest instances (by total
        # annotated pixel area) for the val split; training doesn't need this
        # (it crops from whole images via its own detector, not these masks).
        test_candidates = [
            img_id
            for img_id, cats in test_img_cat_annotations.items()
            if category in cats
        ]
        test_candidates.sort(
            key=lambda img_id, category=category: sum(
                a.get("area", 0)
                for a in test_img_cat_annotations[img_id][category]
            ),
            reverse=True,
        )
        per_category[category] = {
            "train_image_ids": train_candidates[:train_cap],
            "test_image_ids": test_candidates[:test_cap],
        }
    return per_category


def copy_images(
    per_category: dict,
    train_image_id_to_filename: Dict[int, str],
    test_image_id_to_filename: Dict[int, str],
    coco_root: Path,
    output_dir: Path,
) -> None:
    """Copy the selected COCO images: train into one flat train_all/ pool, test
    into val/<category>/. Both folders are rebuilt from scratch."""
    train_all_dir = output_dir / "train_all"
    # Only add-if-missing below; clear stale files from a prior build first so
    # re-running with a different seed/cap/selection logic can't leave old and
    # new selections mixed together in the same flat pool.
    if train_all_dir.exists():
        shutil.rmtree(train_all_dir)
    train_all_dir.mkdir(parents=True, exist_ok=True)
    val_dir = output_dir / "val"
    if val_dir.exists():
        shutil.rmtree(val_dir)

    copied_train_files = set()
    for category, split in per_category.items():
        for img_id in split["train_image_ids"]:
            fname = train_image_id_to_filename[img_id]
            if fname in copied_train_files:
                continue
            copied_train_files.add(fname)
            shutil.copy2(
                coco_root / SPLITS["train"] / fname, train_all_dir / fname
            )

        val_cat_dir = val_dir / category
        val_cat_dir.mkdir(parents=True, exist_ok=True)
        for img_id in split["test_image_ids"]:
            fname = test_image_id_to_filename[img_id]
            shutil.copy2(
                coco_root / SPLITS["test"] / fname, val_cat_dir / fname
            )


# ---------------------------------------------------------------- main


def resolve_categories(
    train_json: dict, categories: Optional[str], top_k_common: int
) -> List[str]:
    """The categories to build: --categories, or the top_k_common categories
    with the most train2017 images."""
    if categories and top_k_common:
        raise SystemExit("Use either --categories or --top-k-common, not both.")
    if top_k_common:
        ranked = rank_categories_by_images(train_json)
        print("Most common train2017 categories (images):")
        for name, count in ranked[:top_k_common]:
            print(f"  {name:15s} {count}")
        return [name for name, _ in ranked[:top_k_common]]
    if categories:
        return [c.strip() for c in categories.split(",") if c.strip()]
    return list(DEFAULT_CATEGORIES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument(
        "--categories", default="", help="Comma-separated COCO category names"
    )
    parser.add_argument(
        "--top-k-common",
        type=int,
        default=0,
        help="Use the K categories present in the most train2017 images",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch missing annotations and only the selected images",
    )
    parser.add_argument(
        "--train-cap",
        type=int,
        default=300,
        help="Max train images per category",
    )
    parser.add_argument(
        "--test-cap", type=int, default=50, help="Max test images per category"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    coco_root = args.coco_root.resolve()
    output_dir = args.output_dir.resolve()
    ensure_annotations(coco_root, args.download)

    train_json = load_annotations(annotations_path(coco_root, SPLITS["train"]))
    categories = resolve_categories(
        train_json, args.categories, args.top_k_common
    )
    _, train_image_id_to_filename, train_img_cat_annotations = build_indices(
        train_json, categories
    )

    val_json = load_annotations(annotations_path(coco_root, SPLITS["test"]))
    _, val_image_id_to_filename, val_img_cat_annotations = build_indices(
        val_json, categories
    )

    per_category = select_per_category(
        train_img_cat_annotations,
        val_img_cat_annotations,
        categories,
        seed=args.seed,
        train_cap=args.train_cap,
        test_cap=args.test_cap,
    )

    print("Per-category selection:")
    for category, split in per_category.items():
        print(
            f"  {category:15s}: train={len(split['train_image_ids']):4d}  "
            f"test={len(split['test_image_ids']):3d}"
        )

    ensure_images(
        coco_root,
        SPLITS["train"],
        [
            train_image_id_to_filename[i]
            for s in per_category.values()
            for i in s["train_image_ids"]
        ],
        args.download,
    )
    ensure_images(
        coco_root,
        SPLITS["test"],
        [
            val_image_id_to_filename[i]
            for s in per_category.values()
            for i in s["test_image_ids"]
        ],
        args.download,
    )
    copy_images(
        per_category,
        train_image_id_to_filename,
        val_image_id_to_filename,
        coco_root,
        output_dir,
    )

    # Manifest: everything the mask-crop/grid builder needs, keyed by
    # (category, split) -> [{"image_id", "file_name", "annotations": [...]}]
    manifest = {
        "categories": categories,
        "seed": args.seed,
        "coco_root": str(coco_root),
        "data": {},
    }
    for category, split in per_category.items():
        manifest["data"][category] = {}
        for split_name, ids_key, id_to_fname, cat_ann in (
            (
                "train",
                "train_image_ids",
                train_image_id_to_filename,
                train_img_cat_annotations,
            ),
            (
                "test",
                "test_image_ids",
                val_image_id_to_filename,
                val_img_cat_annotations,
            ),
        ):
            manifest["data"][category][split_name] = [
                {
                    "image_id": img_id,
                    "file_name": id_to_fname[img_id],
                    "annotations": cat_ann[img_id][category],
                }
                for img_id in split[ids_key]
            ]

    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f)
    print(f"\nSaved manifest: {manifest_path}")
    print(f"train_all pool: {output_dir / 'train_all'}")
    print(f"val (per-category): {output_dir / 'val'}")


if __name__ == "__main__":
    main()
