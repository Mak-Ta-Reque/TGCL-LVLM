"""Step 4 samples from crops.json (tgcl.data.JSONDataset)."""

import json
import random

import pytest

from tgcl.data.image_text_dataset import JSONDataset

CROPS = {
    "cat": {
        "a/img1.jpg": {
            "meta": {"image_size": [640, 480], "patch_size": 200},
            "masks_rle": [
                {
                    "rle": {"counts": "x"},
                    "bbox": [1, 2, 3, 4],
                    "is_concept": True,
                },
                {"bbox": [5.7, 6, 7, 8]},  # box only, float coordinates
                {"rle": None},  # neither mask nor box: skipped
                "bad-entry",  # not a segment: skipped
                {"bbox": ["x", 1, 2, 3]},  # unreadable box: skipped
                {"bbox": [1, 2, 3]},  # not a box: skipped
            ],
        },
        "b/img2.png": {"meta": {"image_size": ["w", 2]}, "masks_rle": []},
        "c/img3.png": "not-a-dict",
        "d/img4.jpg": {"masks_rle": [{"rle": {"counts": "y"}}] * 7},
    },
    "cup": {
        "a/img1.jpg": {
            "meta": {},
            "masks_rle": [{"rle": {"counts": "z"}, "is_concept": False}],
        }
    },
}


@pytest.fixture
def crops(tmp_path):
    path = tmp_path / "crops.json"
    path.write_text(json.dumps(CROPS))
    return str(path)


def _dataset(crops, **kwargs):
    kwargs.setdefault("prompt_template", "cgdl_contains")
    random.seed(1)  # as step 4 does (set_seed) before building the dataset
    return JSONDataset(annotation_file=crops, seed=1, **kwargs).data


def test_one_sample_per_usable_segment(crops):
    data = _dataset(crops, data_dir="root")
    assert {k: len(v) for k, v in data.items()} == {"cat": 9, "cup": 1}
    first, second = data["cat"][:2]
    assert first["img_id"] == "img1_mask_0"
    assert first["image"] == "root/a/img1.jpg"
    assert first["seg_mask_rle"] == {"counts": "x"}
    assert first["bbox"] == [1, 2, 3, 4] and first["is_concept"] is True
    assert first["image_size"] == [640, 480] and first["patch_size"] == 200
    assert second["bbox"] == [5, 6, 7, 8] and "seg_mask_rle" not in second


def test_prompt_names_the_concept(crops):
    data = _dataset(crops, data_dir="")
    for tag, samples in data.items():
        for sample in samples:
            assert tag in sample["instruction"]
            assert "[concept]" not in sample["instruction"]
            assert sample["prompt_concept"] == tag


def test_shuffled_prompt_never_names_a_tag_of_the_image(crops):
    data = _dataset(
        crops,
        data_dir="",
        shuffle_concept_prompt=True,
        shuffle_concept_vocab=["dog", "cat", "cup", "car"],
    )
    img1_tags = {"cat", "cup"}
    for samples in data.values():
        for sample in samples:
            assert sample["prompt_concept"] != sample["concept"]
            if sample["image"] == "a/img1.jpg":
                assert sample["prompt_concept"] not in img1_tags


def test_dataset_size_caps_each_concept(crops):
    data = _dataset(crops, data_dir="", dataset_size=3)
    assert {k: len(v) for k, v in data.items()} == {"cat": 3, "cup": 1}


def test_same_seeds_same_samples(crops):
    a = _dataset(crops, data_dir="", dataset_size=3)
    b = _dataset(crops, data_dir="", dataset_size=3)
    assert a == b
