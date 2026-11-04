#!/usr/bin/env python3
"""{image file name: [ground-truth captions]} for COCO 2017 images (concept-word
alignment BERTScore).

    python scripts/build_coco_captions.py \\
        --annotations data/coco_raw/annotations_captions \\
        --out data/coco_common10/captions.json
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def build(annotation_files) -> dict:
    by_file = defaultdict(list)
    for path in annotation_files:
        data = json.load(open(path, encoding="utf-8"))
        names = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data["annotations"]:
            by_file[names[ann["image_id"]]].append(ann["caption"].strip())
    return dict(by_file)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="folder with captions_*2017.json",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    captions = build(sorted(args.annotations.glob("captions_*2017.json")))
    args.out.write_text(json.dumps(captions))
    print(f"{len(captions)} images with captions -> {args.out}")


if __name__ == "__main__":
    main()
