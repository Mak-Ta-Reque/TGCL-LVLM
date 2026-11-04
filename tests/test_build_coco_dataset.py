"""preprocessing/build_coco10_dataset.py: category choice, offline behaviour."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location(
        "build_coco10_dataset",
        REPO / "preprocessing" / "build_coco10_dataset.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COCO = {
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "dining table"},
        {"id": 3, "name": "cat"},
    ],
    "images": [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(1, 6)],
    "annotations": [
        # person in 3 images (twice in image 1), dining table in 2, cat in 2
        {"image_id": 1, "category_id": 1, "area": 5},
        {"image_id": 1, "category_id": 1, "area": 5},
        {"image_id": 2, "category_id": 1, "area": 5},
        {"image_id": 3, "category_id": 1, "area": 5},
        {"image_id": 4, "category_id": 2, "area": 50},
        {"image_id": 5, "category_id": 2, "area": 9},
        {"image_id": 4, "category_id": 3, "area": 1},
        {"image_id": 5, "category_id": 3, "area": 7},
    ],
}


def test_rank_by_number_of_images_not_instances(builder):
    assert builder.rank_categories_by_images(COCO) == [
        ("person", 3),
        ("cat", 2),
        ("dining table", 2),
    ]


def test_resolve_categories(builder):
    assert builder.resolve_categories(COCO, "", 2) == ["person", "cat"]
    assert builder.resolve_categories(COCO, "cat, dining table", 0) == [
        "cat",
        "dining table",
    ]
    assert builder.resolve_categories(COCO, "", 0) == builder.DEFAULT_CATEGORIES
    with pytest.raises(SystemExit):
        builder.resolve_categories(COCO, "cat", 2)


def test_selection_test_split_prefers_large_objects(builder):
    _, _, idx = builder.build_indices(COCO, ["dining table", "cat"])
    sel = builder.select_per_category(
        idx, idx, ["dining table", "cat"], seed=0, train_cap=5, test_cap=1
    )
    assert sel["dining table"]["test_image_ids"] == [4]
    assert sel["cat"]["test_image_ids"] == [5]
    assert sorted(sel["cat"]["train_image_ids"]) == [4, 5]


def test_unknown_category_is_rejected(builder):
    with pytest.raises(ValueError, match="not found"):
        builder.build_indices(COCO, ["unicorn"])


def test_offline_missing_files_fail_clearly(builder, tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        builder.ensure_annotations(tmp_path, download=False)
    with pytest.raises(FileNotFoundError, match="--download"):
        builder.ensure_images(
            tmp_path, "val2017", ["000000000001.jpg"], download=False
        )
    builder.ensure_images(
        tmp_path, "val2017", [], download=False
    )  # nothing needed -> no error
