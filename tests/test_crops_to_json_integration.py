#!/usr/bin/env python3
"""
Integration tests for preprocessing/crops_to_json.py (process_mapping), with
temporary images and stubbed detectors.

RLE encoding is switched off (RLE=0) so the tests do not need pycocotools; the
mask routing and the JSON structure are what is tested.

Run with:
    python -m pytest tests/test_crops_to_json_integration.py
"""

import json
import os
import shutil
import tempfile
from typing import Dict, Tuple
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from tgcl.preprocessing.crops_to_json import process_mapping


def create_test_image(
    path: str,
    width: int = 400,
    height: int = 300,
    color: Tuple[int, int, int] = (128, 128, 128),
):
    from PIL import Image

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.new("RGB", (width, height), color=color).save(path)
    return path


def create_test_folder_structure(
    base_dir: str, tags: Dict[str, int], img_size: Tuple[int, int] = (400, 300)
):
    """{tag: [absolute image paths]} for <base_dir>/<tag>/image_<i>.jpg."""
    result = {}
    for tag, count in tags.items():
        result[tag] = [
            create_test_image(
                os.path.join(base_dir, tag, f"image_{i:03d}.jpg"), *img_size
            )
            for i in range(count)
        ]
    return result


def create_test_mapping_json(
    json_path: str,
    input_root: str,
    tags: Dict[str, int],
    img_size: Tuple[int, int] = (400, 300),
):
    """Write {tag: [relative paths]} and the images; returns the mapping."""
    mapping = {}
    for tag, count in tags.items():
        mapping[tag] = []
        for i in range(count):
            rel_path = f"{tag}/img_{i:03d}.png"
            create_test_image(os.path.join(input_root, rel_path), *img_size)
            mapping[tag].append(rel_path)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return mapping


def two_masks(height: int = 300, width: int = 400):
    mask_one = np.zeros((height, width), dtype=bool)
    mask_one[20:80, 10:60] = True
    mask_two = np.zeros((height, width), dtype=bool)
    mask_two[140:200, 220:300] = True
    return [([10, 20, 50, 60], mask_one), ([220, 140, 80, 60], mask_two)]


class _TempDirCase(unittest.TestCase):
    """A test case with a fresh temporary directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="crops_test_")
        self.input_dir = os.path.join(self.temp_dir, "images")
        self.mapping_json = os.path.join(self.temp_dir, "mapping.json")
        self.output_json = os.path.join(self.temp_dir, "crops.json")
        self.env = patch.dict(os.environ, {"RLE": "0"}, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def load_output(self):
        with open(self.output_json, encoding="utf-8") as handle:
            return json.load(handle)


@patch(
    "tgcl.preprocessing.crops_to_json.detect_auto_masks",
    side_effect=lambda images, **kw: [two_masks() for _ in images],
)
@patch(
    "tgcl.preprocessing.crops_to_json.detect_concept_masks",
    side_effect=lambda images, **kw: [two_masks()[:1] for _ in images],
)
@patch(
    "tgcl.preprocessing.crops_to_json._load_langsam", return_value=MagicMock()
)
class TestProcessMapping(_TempDirCase):
    """Tag filtering, capping, output structure and determinism (langsam
    detector, stubbed)."""

    def run_mapping(self, **kwargs):
        args = dict(
            detector="langsam",
            batch_size=2,
            masks_per_image=3,
            concept_masks_per_image=1,
            min_images_per_tag=1,
            image_size_width=400,
            patch_size=100,
            show_progress=False,
            seed=0,
        )
        args.update(kwargs)
        with patch.dict(os.environ, {"CROP_MODE": "langsam"}, clear=False):
            process_mapping(
                self.mapping_json, self.input_dir, self.output_json, **args
            )
        return self.load_output()

    def test_output_structure(self, *_mocks):
        create_test_mapping_json(
            self.mapping_json, self.input_dir, {"apple": 2}
        )
        result = self.run_mapping()
        self.assertEqual(
            sorted(result["apple"]), ["apple/img_000.png", "apple/img_001.png"]
        )
        entry = result["apple"]["apple/img_000.png"]
        self.assertEqual(entry["meta"]["image_size"], [400, 300])
        self.assertEqual(entry["meta"]["concept_mask_index"], 0)
        masks = entry["masks_rle"]
        self.assertTrue(masks[0]["is_concept"])
        self.assertTrue(all(not m["is_concept"] for m in masks[1:]))
        for m in masks:
            x, y, w, h = m["bbox"]
            self.assertTrue(0 <= x and 0 <= y and x + w <= 400 and y + h <= 300)

    def test_min_images_per_tag_filter(self, *_mocks):
        create_test_mapping_json(
            self.mapping_json, self.input_dir, {"apple": 3, "pear": 1}
        )
        result = self.run_mapping(min_images_per_tag=2)
        self.assertEqual(list(result), ["apple"])

    def test_max_images_per_tag_cap_is_seeded(self, *_mocks):
        create_test_mapping_json(
            self.mapping_json, self.input_dir, {"apple": 6}
        )
        first = sorted(self.run_mapping(max_images_per_tag=3, seed=7)["apple"])
        second = sorted(self.run_mapping(max_images_per_tag=3, seed=7)["apple"])
        self.assertEqual(len(first), 3)
        self.assertEqual(first, second)

    def test_no_surviving_tag_raises(self, *_mocks):
        create_test_mapping_json(
            self.mapping_json, self.input_dir, {"apple": 1}
        )
        with self.assertRaises(RuntimeError):
            self.run_mapping(min_images_per_tag=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
