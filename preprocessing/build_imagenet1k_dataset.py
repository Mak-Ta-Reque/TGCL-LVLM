#!/usr/bin/env python3
"""ImageNet-1k subset from the Hugging Face dataset ILSVRC/imagenet-1k (gated:
HF_TOKEN must belong to an account that accepted the ImageNet license).

Writes, under --output-dir (default data/imagenet1k):
  train_all/<tag_with_underscores>__<n>.JPEG   --train-per-class images per
  class (pipeline INPUT_DIR)
  val/<tag_with_underscores>/<n>.JPEG          --test-per-class validation
  images per class
  vocab.txt                                    one class tag per line
  (CONCEPTS_VOCAB)
  concepts_to_images.json                      tag -> its train_all files
  (pipeline CONCEPT_MAP)
  manifest.json                                class index -> synset, tag,
  folder; counts; seed

Class tags are the first ImageNet name of each class, lower-case, letters and
spaces only (hyphens become spaces), so the pipeline's concept-name cleaning
leaves them unchanged. The three names shared by two classes (crane, cardigan,
maillot) use TAG_OVERRIDES; any other collision would get a synonym appended.

Train shards are read in a seeded random order and deleted after use, until
every class has --train-per-class images (about 40% of the 146 GB); the 50k
validation images (50 per class) are all used with the default --test-per-class.

Usage:
    python preprocessing/build_imagenet1k_dataset.py --train-per-class 300 \\
        --test-per-class 50
    python preprocessing/create_grids.py --input_dir data/imagenet1k/val \\
        --output_dir data/imagenet1k/val_grids --n 4 --num_grids 500
        --image_size 512 --seed 0
"""

import argparse
import ast
from collections import Counter
import io
import json
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple

from PIL import Image

REPO_ID = "ILSVRC/imagenet-1k"
# The three ImageNet names shared by two classes, as words the VLM can say.
TAG_OVERRIDES = {
    "n02012849": "crane bird",
    "n03126707": "construction crane",
    "n02113186": "cardigan welsh corgi",
    "n02963159": "cardigan sweater",
    "n03710637": "maillot",
    "n03710721": "tank suit",
}


def parse_classes(source: str) -> List[Tuple[str, str]]:
    """[(wnid, "name, synonym, ..."), ...] in label order, from the dataset's
    classes.py (parsed, not run)."""
    start = source.index("{", source.index("IMAGENET2012_CLASSES"))
    depth = 0
    for i in range(start, len(source)):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            break
    return list(ast.literal_eval(source[start : i + 1]).items())


def clean_tag(name: str) -> str:
    return re.sub(
        r"\s+", " ", re.sub(r"[^a-z ]", "", name.lower().replace("-", " "))
    ).strip()


def class_tags(classes: List[Tuple[str, str]]) -> List[str]:
    """One unique, cleaned tag per class (label order): TAG_OVERRIDES, else the
    first name."""
    first = [
        TAG_OVERRIDES.get(wnid) or clean_tag(names.split(",")[0])
        for wnid, names in classes
    ]
    counts = Counter(first)
    tags = []
    for (wnid, names), tag in zip(classes, first):
        if counts[tag] > 1:
            synonyms = [clean_tag(s) for s in names.split(",")[1:]]
            extra = next((s for s in synonyms if s and s != tag), wnid)
            tag = f"{tag} {extra}"
        tags.append(tag)
    duplicates = [t for t, c in Counter(tags).items() if c > 1]
    if duplicates:
        raise ValueError(f"class tags still not unique: {duplicates}")
    return tags


def _download(filename: str, raw_dir: Path, token: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            REPO_ID,
            filename,
            repo_type="dataset",
            token=token,
            local_dir=raw_dir,
        )
    )


def _rows(parquet: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["image", "label"])
    for image, label in zip(
        table.column("image").to_pylist(), table.column("label").to_pylist()
    ):
        yield image["bytes"], int(label)


def _save(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(data)).convert("RGB").save(path, "JPEG", quality=95)


def fill(
    split_files: List[str],
    per_class: int,
    n_classes: int,
    out_path,
    raw_dir: Path,
    token: str,
    keep_raw: bool,
) -> Dict[int, int]:
    """Take up to per_class images of every class from the shards, in order;
    stop when all are full."""
    counts: Dict[int, int] = Counter()
    for shard in split_files:
        if len(counts) == n_classes and min(counts.values()) >= per_class:
            break
        path = _download(shard, raw_dir, token)
        for data, label in _rows(path):
            if 0 <= label < n_classes and counts[label] < per_class:
                _save(data, out_path(label, counts[label]))
                counts[label] += 1
        if not keep_raw:
            path.unlink()
        full = sum(c >= per_class for c in counts.values())
        print(
            f"{shard}: {full}/{n_classes} classes full, "
            f"{sum(counts.values())} images",
            flush=True,
        )
    return counts


def write_concept_map(out: Path, target: Path = None) -> Path:
    """concepts_to_images.json: tag -> sorted train_all file names (pipeline
    CONCEPT_MAP). Tags contain no '_', so the file prefix <folder> maps back by
    '_' -> ' '."""
    mapping: Dict[str, List[str]] = {}
    for path in sorted((out / "train_all").glob("*.JPEG")):
        mapping.setdefault(
            path.name.rsplit("__", 1)[0].replace("_", " "), []
        ).append(path.name)
    target = target or out / "concepts_to_images.json"
    target.write_text(json.dumps(mapping, indent=1))
    return target


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output-dir", type=Path, default=Path("data/imagenet1k"))
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/imagenet1k_raw"),
        help="shard download folder",
    )
    ap.add_argument("--train-per-class", type=int, default=300)
    ap.add_argument("--test-per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--keep-raw", action="store_true", help="keep downloaded shards"
    )
    ap.add_argument(
        "--concept-map-only",
        action="store_true",
        help="only (re)write concepts_to_images.json from an existing "
        "train_all/",
    )
    args = ap.parse_args()
    if args.concept_map_only:
        print("wrote", write_concept_map(args.output_dir))
        return

    from huggingface_hub import hf_hub_download
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    files = sorted(
        s.rfilename for s in HfApi(token=token).dataset_info(REPO_ID).siblings
    )
    classes = parse_classes(
        Path(
            hf_hub_download(
                REPO_ID,
                "classes.py",
                repo_type="dataset",
                token=token,
                local_dir=args.raw_dir,
            )
        ).read_text(encoding="utf-8")
    )
    tags = class_tags(classes)
    folders = [t.replace(" ", "_") for t in tags]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.txt").write_text("\n".join(tags) + "\n")

    val_files = [f for f in files if f.startswith("data/validation-")]
    val_counts = fill(
        val_files,
        args.test_per_class,
        len(tags),
        lambda label, i: out / "val" / folders[label] / f"{i:04d}.JPEG",
        args.raw_dir,
        token,
        args.keep_raw,
    )
    train_files = [f for f in files if f.startswith("data/train-")]
    random.Random(args.seed).shuffle(train_files)
    train_counts = fill(
        train_files,
        args.train_per_class,
        len(tags),
        lambda label, i: out / "train_all" / f"{folders[label]}__{i:04d}.JPEG",
        args.raw_dir,
        token,
        args.keep_raw,
    )

    manifest = {
        "source": REPO_ID,
        "seed": args.seed,
        "train_per_class": args.train_per_class,
        "test_per_class": args.test_per_class,
        "classes": [
            {
                "label": i,
                "wnid": wnid,
                "tag": tag,
                "folder": folder,
                "train": train_counts.get(i, 0),
                "test": val_counts.get(i, 0),
            }
            for i, ((wnid, _), tag, folder) in enumerate(
                zip(classes, tags, folders)
            )
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    write_concept_map(out)
    short = [
        c["tag"]
        for c in manifest["classes"]
        if c["train"] < args.train_per_class or c["test"] < args.test_per_class
    ]
    print(
        f"done: {sum(train_counts.values())} train, "
        f"{sum(val_counts.values())} test images; "
        f"{len(short)} classes short of the "
        f"target{': ' + ', '.join(short[:10]) if short else ''}"
    )


if __name__ == "__main__":
    main()
