"""Class-free segmentation switches in preprocessing/crops_to_json.py.

C3: CONCEPT_MASKS_PER_IMAGE=0 -> automatic masks only, the tag text is never
    sent to the detector.
C5: SLIDING_WINDOW_TAG_FILTER=0 -> sliding windows are not ranked by the tag.
Models are replaced by stubs; only the routing logic is under test.
"""

import json

import numpy as np
from PIL import Image
import pytest

from tgcl.preprocessing import crops_to_json

SIZE = 64


@pytest.fixture
def mapping(tmp_path):
    root = tmp_path / "images"
    (root / "cat").mkdir(parents=True)
    rels = []
    for i in range(2):
        rel = f"cat/{i}.jpg"
        Image.new("RGB", (SIZE, SIZE), (40 * i, 80, 120)).save(root / rel)
        rels.append(rel)
    mapping_json = tmp_path / "mapping.json"
    mapping_json.write_text(json.dumps({"cat": rels}))
    return str(mapping_json), str(root), tmp_path / "crops.json"


def _stripe_masks(n):
    """n non-overlapping horizontal stripes, each a valid (bbox, mask) pair."""
    pairs = []
    height = SIZE // n
    for k in range(n):
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        mask[k * height : (k + 1) * height, :] = True
        pairs.append(([0, k * height, SIZE, height], mask))
    return pairs


def test_class_free_masks_skip_tag_detection(mapping, monkeypatch):
    mapping_json, root, out = mapping
    requested = {}

    def no_tag_detection(*args, **kwargs):
        raise AssertionError(
            "tag-prompted detection must not run in class-free mode"
        )

    def auto_masks(pil_images, detector, model, topn, min_mask_area):
        requested["topn"] = topn
        return [_stripe_masks(topn) for _ in pil_images]

    monkeypatch.setattr(
        crops_to_json, "_load_langsam", lambda device=None: object()
    )
    monkeypatch.setattr(crops_to_json, "detect_concept_masks", no_tag_detection)
    monkeypatch.setattr(crops_to_json, "detect_auto_masks", auto_masks)
    monkeypatch.setenv("CROP_MODE", "langsam")
    monkeypatch.setenv(
        "RLE", "0"
    )  # mask routing only; RLE encoding needs pycocotools

    crops_to_json.process_mapping(
        mapping_json,
        root,
        str(out),
        detector="langsam",
        masks_per_image=4,
        concept_masks_per_image=0,
        min_images_per_tag=1,
        image_size_width=SIZE,
        show_progress=False,
        seed=0,
    )

    assert requested["topn"] == 4  # no slot reserved for a tag mask
    entries = json.loads(out.read_text())["cat"]
    assert len(entries) == 2
    for entry in entries.values():
        masks = entry["masks_rle"]
        assert len(masks) == 4
        assert not any(m["is_concept"] for m in masks)
        assert "concept_mask_index" not in entry["meta"]


def test_class_free_masks_reject_positive_negative_mode(mapping, monkeypatch):
    mapping_json, root, out = mapping
    monkeypatch.setattr(
        crops_to_json, "_load_langsam", lambda device=None: object()
    )
    with pytest.raises(ValueError, match="positive_negative_segment"):
        crops_to_json.process_mapping(
            mapping_json,
            root,
            str(out),
            detector="langsam",
            concept_masks_per_image=0,
            min_images_per_tag=1,
            positive_negative_segment=True,
            show_progress=False,
        )


@pytest.mark.parametrize(
    "flag, expected_tag", [(None, "cat"), ("1", "cat"), ("0", None)]
)
def test_sliding_window_tag_filter_switch(
    mapping, monkeypatch, flag, expected_tag
):
    mapping_json, root, out = mapping
    seen_tags = []

    def windows(image, size, count, *clip, tag=None, **kwargs):
        seen_tags.append(tag)
        return [([0, 0, 16, 16], None)]

    monkeypatch.setattr(
        crops_to_json,
        "_load_clip_model",
        lambda device=None: (None, None, "cpu"),
    )
    monkeypatch.setattr(
        crops_to_json, "_build_sliding_window_pairs_for_image", windows
    )
    monkeypatch.setenv("CROP_MODE", "sliding_window")
    if flag is None:
        monkeypatch.delenv("SLIDING_WINDOW_TAG_FILTER", raising=False)
    else:
        monkeypatch.setenv("SLIDING_WINDOW_TAG_FILTER", flag)

    crops_to_json.process_mapping(
        mapping_json,
        root,
        str(out),
        detector="none",
        masks_per_image=1,
        min_images_per_tag=1,
        image_size_width=SIZE,
        show_progress=False,
        seed=0,
    )
    assert seen_tags == [expected_tag, expected_tag]
