"""CPU units of steps 1-2: masks, inpainting, concept maps, explanations."""

import copy
import csv
import json

import numpy as np
from PIL import Image
import pytest

from tgcl.helpers.explanations import compact
from tgcl.helpers.explanations import expand
from tgcl.imputers import noisy_linear_inpaint
from tgcl.mask_utils import decode_mask_rle
from tgcl.mask_utils import encode_mask_rle
from tgcl.preprocessing.concept_image_mapping import process_csv_file
from tgcl.preprocessing.concept_image_mapping import sanitize_concept
from tgcl.preprocessing.select_top_concepts import select_top_concepts

# ------------------------------------------------------------- masks -----


def test_rle_round_trip_is_exact_and_json_safe():
    rng = np.random.default_rng(0)
    mask = rng.random((37, 53)) > 0.6
    rle = encode_mask_rle(mask)
    assert isinstance(rle["counts"], str) and rle["size"] == [37, 53]
    json.dumps(rle)
    assert np.array_equal(decode_mask_rle(json.loads(json.dumps(rle))), mask)


# -------------------------------------------------------- inpainting -----


def test_inpainting_keeps_the_foreground_and_fills_the_background():
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[:, :12] = 200  # left half bright, right half dark
    keep = np.zeros((24, 24), dtype=bool)
    keep[:, :12] = True
    out = np.asarray(noisy_linear_inpaint(Image.fromarray(img), keep, 0.0))
    assert out.shape == img.shape
    assert np.abs(out[:, :12].astype(int) - 200).max() <= 1
    assert out[:, 12:].mean() > 50  # filled from the kept side, not left dark


# ------------------------------------------------------- concept maps -----


def _write_objects_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "predicted_text"])
        w.writeheader()
        w.writerows(rows)


def test_concept_map_from_answers(tmp_path):
    objects = tmp_path / "objects.csv"
    _write_objects_csv(
        objects,
        [
            {"image_name": "a.jpg", "predicted_text": "Cars, a dog, the tree"},
            {"image_name": "b.jpg", "predicted_text": "dog, 3 cups, café"},
            {"image_name": "c.jpg", "predicted_text": ""},
        ],
    )
    mapping = process_csv_file(objects, tmp_path / "map.json")
    assert mapping["car"] == ["a.jpg"]
    assert mapping["dog"] == ["a.jpg", "b.jpg"]
    assert mapping["tree"] == ["a.jpg"]
    assert json.loads((tmp_path / "map.json").read_text()) == mapping


@pytest.mark.parametrize(
    "raw, kept", [("dog", "dog"), ("the", None), ("café", None), ("", None)]
)
def test_sanitize_concept(raw, kept):
    assert sanitize_concept(raw) == kept


def test_top_concepts_by_image_count_within_the_vocab(tmp_path):
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps({"dog": ["1", "2", "3"], "cat": ["1"], "sky": ["1", "2"]})
    )
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("dog\ncat\n")
    out = select_top_concepts(mapping, 1, vocab)
    assert out.name == "map_vocab_top1.json"
    assert json.loads(out.read_text()) == {"dog": ["1", "2", "3"]}


def test_top_concepts_file_is_rewritten_for_a_new_map(tmp_path):
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"dog": ["1", "2"], "cat": ["1"]}))
    first = select_top_concepts(mapping, 1)
    mapping.write_text(json.dumps({"dog": ["1"], "cat": ["1", "2"]}))
    second = select_top_concepts(mapping, 1)
    assert first == second
    assert json.loads(second.read_text()) == {"cat": ["1", "2"]}


def test_no_cap_and_no_vocab_keeps_the_map(tmp_path):
    mapping = tmp_path / "map.json"
    mapping.write_text("{}")
    assert select_top_concepts(mapping, 0) == mapping


# ------------------------------------------------ explanation tables -----


def _payload():
    concept = {
        "concept_name": ["dog"],
        "text_grounding": ["dog", "puppy"],
        "image_grounding_path": ["a.jpg"],
        "image_grounding_bboxes": [[0, 0, 4, 4]],
        "image_grounding_masks": ["rle"],
    }
    entry = {"rank": 1, "concept_index": 3, "similarity": 0.9, **concept}
    return {
        "results": [
            {
                "per_token_concepts": [{"top_concepts": [dict(entry)]}],
                "top_concepts_over_sequence": [dict(entry)],
            }
        ]
    }


def test_compact_then_expand_restores_the_entries_without_masks():
    original = _payload()
    payload = compact(copy.deepcopy(original))
    token_entry = payload["results"][0]["per_token_concepts"][0][
        "top_concepts"
    ][0]
    assert set(token_entry) == {"rank", "concept_index", "similarity"}
    assert "image_grounding_masks" not in payload["concept_table"]["3"]
    restored = expand(payload)["results"][0]["top_concepts_over_sequence"][0]
    expected = dict(original["results"][0]["top_concepts_over_sequence"][0])
    del expected["image_grounding_masks"]
    assert restored == expected


def test_compact_rejects_two_groundings_for_one_concept():
    payload = _payload()
    payload["results"][0]["top_concepts_over_sequence"][0]["concept_name"] = [
        "cat"
    ]
    with pytest.raises(ValueError):
        compact(payload)
