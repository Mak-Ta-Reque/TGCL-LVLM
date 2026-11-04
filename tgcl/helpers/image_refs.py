"""Evaluation images as references: nothing derived from an image (crop, grid,
resized copy) is written to disk; it is built in memory from the reference when
needed.

A reference is a string:

    <path>                        an image file; its class is its folder name
    (data/<set>/val/<class>/x.jpg)
                                  or, for a legacy grid file
                                  grid_<n>_<i>__<a>__<b>...jpg, the names in it
    grid:<spec.json>#<i>          grid i of a grid spec, composed in memory from
    its cells (paths or refs)
    crop:<spec.json>#<i>          crop i of a crop spec: a box (+ context) cut
    in memory from an image file

A list of references comes from a directory (every image below it) or a JSON
file:

    {"images": ["data/imagenet1k/val/acorn/0000.JPEG", ...]}
    {"cell_size": 384, "grids": [{"cells": [<path or ref>, ...], "labels":
    ["acorn", ...]}, ...]}
    {"context": 0.1, "crops": [{"image": <path>, "bbox": [x, y, w, h], "label":
    "car"}, ...]}

Crop boxes are in the image's own pixels; context is the margin added on every
side, as a fraction of the box's longer side. Relative paths are relative to the
working directory (the repo root), as elsewhere.
"""

from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

__all__ = [
    "GRID_PREFIX",
    "CROP_PREFIX",
    "IMAGE_EXTENSIONS",
    "is_grid_ref",
    "is_crop_ref",
    "is_memory_ref",
    "grid_ref",
    "crop_ref",
    "list_refs",
    "load_image",
    "image_classes",
    "ref_name",
    "source_image",
]

GRID_PREFIX = "grid:"
CROP_PREFIX = "crop:"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_grid_ref(ref: Union[str, Path, None]) -> bool:
    return isinstance(ref, str) and ref.startswith(GRID_PREFIX)


def is_crop_ref(ref: Union[str, Path, None]) -> bool:
    return isinstance(ref, str) and ref.startswith(CROP_PREFIX)


def is_memory_ref(ref: Union[str, Path, None]) -> bool:
    """A reference whose image is built in memory (grid or crop), not an image
    file."""
    return is_grid_ref(ref) or is_crop_ref(ref)


def grid_ref(spec: Union[str, Path], index: int) -> str:
    return f"{GRID_PREFIX}{os.path.abspath(spec)}#{index}"


def crop_ref(spec: Union[str, Path], index: int) -> str:
    return f"{CROP_PREFIX}{os.path.abspath(spec)}#{index}"


def _parse_ref(ref: str):
    prefix = GRID_PREFIX if is_grid_ref(ref) else CROP_PREFIX
    spec, _, index = ref[len(prefix) :].rpartition("#")
    return spec, int(index)


@lru_cache(maxsize=16)
def _load_spec(spec: str) -> dict:
    with open(spec, encoding="utf-8") as f:
        data = json.load(f)
    for i, grid in enumerate(data.get("grids", [])):
        cells, labels = grid.get("cells", []), grid.get("labels", [])
        side = math.isqrt(len(cells))
        if not cells or side * side != len(cells) or len(labels) != len(cells):
            raise ValueError(
                f"{spec}: grid {i} needs a square number of cells with "
                "one label each"
            )
    for i, crop in enumerate(data.get("crops", [])):
        box = crop.get("bbox") or []
        if (
            not crop.get("image")
            or not crop.get("label")
            or len(box) != 4
            or min(box[2], box[3]) <= 0
        ):
            raise ValueError(
                f"{spec}: crop {i} needs an image, a label and a box [x, "
                "y, w, h] with w, h > 0"
            )
    return data


def list_refs(source: Union[str, Path]) -> List[str]:
    """The references of a directory (sorted image files below it) or of a JSON
    list file."""
    source = Path(source)
    if source.is_dir():
        return sorted(
            str(p.resolve())
            for p in source.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if source.suffix.lower() != ".json":
        raise ValueError(
            f"{source}: expected a directory or a .json image list"
        )
    data = _load_spec(str(source.resolve()))
    if "grids" in data:
        return [grid_ref(source, i) for i in range(len(data["grids"]))]
    if "crops" in data:
        return [crop_ref(source, i) for i in range(len(data["crops"]))]
    return [str(Path(p).resolve()) for p in data["images"]]


def _crop_box(width: int, height: int, bbox, context: float):
    x, y, w, h = (float(v) for v in bbox)
    pad = context * max(w, h)
    x1, y1 = max(0, int(round(x - pad))), max(0, int(round(y - pad)))
    x2, y2 = min(width, int(round(x + w + pad))), min(
        height, int(round(y + h + pad))
    )
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def load_image(ref: Union[str, Path, Image.Image]) -> Image.Image:
    """The RGB image a reference stands for (a grid or crop is built here, in
    memory)."""
    if isinstance(ref, Image.Image):
        return ref.convert("RGB")
    if is_crop_ref(ref):
        spec, index = _parse_ref(ref)
        data = _load_spec(spec)
        crop = data["crops"][index]
        with Image.open(crop["image"]) as im:
            img = im.convert("RGB")
        return img.crop(
            _crop_box(
                img.width,
                img.height,
                crop["bbox"],
                float(crop.get("context", data.get("context", 0.0))),
            )
        )
    if not is_grid_ref(ref):
        with Image.open(ref) as im:
            return im.convert("RGB")
    spec, index = _parse_ref(ref)
    data = _load_spec(spec)
    grid = data["grids"][index]
    cell = int(grid.get("cell_size", data.get("cell_size", 384)))
    side = math.isqrt(len(grid["cells"]))
    canvas = Image.new("RGB", (side * cell, side * cell))
    for k, cell_ref in enumerate(grid["cells"]):
        canvas.paste(
            load_image(cell_ref).resize((cell, cell)),
            ((k % side) * cell, (k // side) * cell),
        )
    return canvas


def image_classes(ref: Optional[Union[str, Path]]) -> List[str]:
    """True class name(s) of an evaluation image ('_' -> ' '): a grid's labels
    in cell order, a crop's label, else the image's class folder."""
    if not ref:
        return []
    if is_grid_ref(ref):
        spec, index = _parse_ref(ref)
        return [
            c.replace("_", " ")
            for c in _load_spec(spec)["grids"][index]["labels"]
        ]
    if is_crop_ref(ref):
        spec, index = _parse_ref(ref)
        return [_load_spec(spec)["crops"][index]["label"].replace("_", " ")]
    path = Path(ref)
    if path.name.startswith("grid_") and "__" in path.stem:
        return [c.replace("_", " ") for c in path.stem.split("__")[1:]]
    return [path.parent.name.replace("_", " ")]


def source_image(ref: Union[str, Path]) -> Optional[str]:
    """The image file a reference shows (a crop's image; None for a grid): e.g.
    to find its captions."""
    if is_grid_ref(ref):
        return None
    if is_crop_ref(ref):
        spec, index = _parse_ref(ref)
        return str(_load_spec(spec)["crops"][index]["image"])
    return str(ref)


def ref_name(ref: Union[str, Path]) -> str:
    """A short stable name for a reference (file name, grid_<i>__<labels>,
    crop_<i>__<image>)."""
    if is_grid_ref(ref):
        _, index = _parse_ref(ref)
        return f"grid_{index}__" + "__".join(
            c.replace(" ", "_") for c in image_classes(ref)
        )
    if is_crop_ref(ref):
        _, index = _parse_ref(ref)
        return f"crop_{index}__{Path(source_image(ref)).name}"
    return Path(ref).name
