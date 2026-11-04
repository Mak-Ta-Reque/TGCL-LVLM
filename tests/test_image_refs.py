"""helpers.image_refs and preprocessing/build_subset.py: evaluation images as
JSON references, grids composed in memory, no image files written."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tgcl.helpers import image_refs

REPO = Path(__file__).resolve().parent.parent

COLORS = {
    "acorn": (200, 0, 0),
    "school_bus": (0, 200, 0),
    "toucan": (0, 0, 200),
    "goldfish": (200, 200, 0),
}


def _dataset(root: Path) -> Path:
    """data/<set>: val/<class folder>/<n>.png in one color per class, plus a
    concept map."""
    for folder, color in COLORS.items():
        (root / "val" / folder).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (30 + i, 20), color).save(
                root / "val" / folder / f"{i:04d}.png"
            )  # lossless
    names = [f.replace("_", " ") for f in COLORS]
    json.dump(
        {n: [f"{n}__{i}.JPEG" for i in range(5)] for n in names},
        open(root / "concepts_to_images.json", "w", encoding="utf-8"),
    )
    return root


def test_path_refs_and_classes(tmp_path):
    root = _dataset(tmp_path / "set")
    refs = image_refs.list_refs(root / "val")
    assert len(refs) == 12 and all(Path(r).is_absolute() for r in refs)
    assert image_refs.image_classes(
        str(root / "val" / "school_bus" / "0000.png")
    ) == ["school bus"]
    assert (
        image_refs.image_classes(
            "grid_4_7__acorn__school_bus__toucan__goldfish.jpg"
        )[1]
        == "school bus"
    )
    assert image_refs.load_image(refs[0]).mode == "RGB"


def test_grid_spec_composes_in_memory(tmp_path):
    root = _dataset(tmp_path / "set")
    cells = [str(root / "val" / f / "0001.png") for f in COLORS]
    spec = tmp_path / "grids.json"
    json.dump(
        {"cell_size": 16, "grids": [{"cells": cells, "labels": list(COLORS)}]},
        open(spec, "w", encoding="utf-8"),
    )
    before = set(tmp_path.rglob("*"))
    (ref,) = image_refs.list_refs(spec)
    assert image_refs.is_grid_ref(ref)
    assert image_refs.image_classes(ref) == [
        "acorn",
        "school bus",
        "toucan",
        "goldfish",
    ]
    img = np.asarray(image_refs.load_image(ref))
    assert img.shape == (32, 32, 3)
    assert (
        tuple(img[8, 8]) == COLORS["acorn"]
        and tuple(img[24, 24]) == COLORS["goldfish"]
    )  # cell order
    assert (
        image_refs.ref_name(ref)
        == "grid_0__acorn__school_bus__toucan__goldfish"
    )
    assert set(tmp_path.rglob("*")) == before  # nothing written


def test_build_subset_writes_json_only(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_subset", REPO / "preprocessing" / "build_subset.py"
    )
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    root = _dataset(tmp_path / "set")
    files = bs.build(
        root,
        ["acorn", "school bus", "toucan", "goldfish"],
        num_grids=3,
        grid_n=4,
        cell_size=16,
        seed=0,
    )
    assert set(files) == {
        "concepts_to_images.json",
        "vocab.txt",
        "test.json",
        "grids.json",
    }
    assert files["vocab.txt"].split() == [
        "acorn",
        "school",
        "bus",
        "toucan",
        "goldfish",
    ]
    assert len(files["test.json"]["images"]) == 12
    for grid in files["grids.json"]["grids"]:
        assert len(set(grid["labels"])) == 4  # distinct classes per grid
        for cell, label in zip(grid["cells"], grid["labels"]):
            assert Path(cell).parent.name == label.replace(
                " ", "_"
            )  # cell matches its label
    only = bs.build(
        root, [], num_grids=1, grid_n=4, cell_size=16, seed=0, grids_only=True
    )
    assert set(only) == {"test.json", "grids.json"}


def test_crop_refs_cut_in_memory_with_context(tmp_path):
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 50:90] = (200, 0, 0)  # the "object": 40 x 20 px at (50, 40)
    path = tmp_path / "scene.png"
    Image.fromarray(img).save(path)
    spec = tmp_path / "test.json"
    json.dump(
        {
            "context": 0.25,
            "crops": [
                {
                    "image": str(path),
                    "bbox": [50, 40, 40, 20],
                    "label": "dining_table",
                }
            ],
        },
        open(spec, "w", encoding="utf-8"),
    )
    before = set(tmp_path.rglob("*"))
    (ref,) = image_refs.list_refs(spec)
    assert image_refs.is_crop_ref(ref) and image_refs.is_memory_ref(ref)
    crop = image_refs.load_image(ref)
    assert crop.size == (60, 40)  # 40 + 2 * 10 (0.25 x 40) context per side
    assert tuple(np.asarray(crop)[20, 30]) == (
        200,
        0,
        0,
    )  # object in the middle
    assert image_refs.image_classes(ref) == ["dining table"]
    assert image_refs.source_image(ref) == str(
        path
    )  # its captions are the image's
    assert image_refs.ref_name(ref) == "crop_0__scene.png"
    grids = tmp_path / "grids.json"
    json.dump(
        {
            "cell_size": 8,
            "grids": [{"cells": [ref] * 4, "labels": ["dining table"] * 4}],
        },
        open(grids, "w", encoding="utf-8"),
    )
    (grid,) = image_refs.list_refs(grids)
    assert image_refs.load_image(grid).size == (16, 16)  # grid of crops
    assert image_refs.source_image(grid) is None
    assert set(tmp_path.rglob("*")) == before | {
        grids
    }  # nothing but the JSON written


def test_detection_keeps_the_best_large_enough_box():
    spec = importlib.util.spec_from_file_location(
        "build_detected_test_set",
        REPO / "preprocessing" / "build_detected_test_set.py",
    )
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    result = {
        "boxes": [[0, 0, 10, 10], [5, 5, 105, 55], [0, 0, 300, 200]],
        "scores": [0.9, 0.6, 0.4],
    }
    assert bd.best_detection(result, min_side=16) == {
        "bbox": [5.0, 5.0, 100.0, 50.0],
        "score": 0.6,
    }
    assert bd.best_detection({"boxes": [], "scores": []}, min_side=16) is None
