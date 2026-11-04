"""scripts/report_metrics.py (Obj-F1): label matching and macro F1."""

import json

import pytest

import tgcl.reporting.report_metrics as objf1


def _explanations(path, items):
    results = [
        {
            "image_path": f"/eval/{folder}/{i}.jpg",
            "per_token_concepts": [
                {
                    "top_concepts": [
                        {"rank": 1, "concept_name": [pred1]},
                        {"rank": 2, "concept_name": [pred2]},
                    ]
                }
            ],
        }
        for i, (folder, pred1, pred2) in enumerate(items)
    ]
    path.write_text(json.dumps({"results": results}))


def test_folder_and_concept_names_match_across_separators(tmp_path):
    path = tmp_path / "vlm_explanations.json"
    _explanations(
        path,
        [
            ("garbage_truck", "garbage truck", "dog"),
            ("Dining table", "dining table", "cup"),
        ],
    )
    rows = objf1.load_predictions_from_json(path)
    assert rows == [
        ("garbage truck", "garbage truck", "dog"),
        ("dining table", "dining table", "cup"),
    ]
    metrics = objf1.compute_metrics(
        rows, classes=["garbage_truck", "dining table"]
    )
    assert metrics["top1_accuracy"] == 1.0 and metrics["top1_f1_macro"] == 1.0


def test_macro_f1_by_hand():
    # classes a, b, c; c never occurs -> F1 0 counts in the top-1 macro mean
    # (sklearn
    # labels=..., zero_division=0) but is skipped by the top-2 containment F1.
    rows = [
        ("a", "a", "b"),
        ("a", "b", "a"),
        ("b", "b", None),
        ("b", None, None),
    ]
    m = objf1.compute_metrics(rows, classes=["a", "b", "c"])
    # top-1: a: tp1 fp0 fn1 -> 2/3 ; b: tp1 fp1 fn1 -> 1/2 ; c: 0
    assert m["top1_f1_macro"] == pytest.approx((2 / 3 + 1 / 2 + 0) / 3)
    # top-2: a: tp2 fp0 fn0 -> 1 ; b: tp1 fp2 fn1 -> P 1/3, R 1/2 -> 0.4
    assert m["top2_f1_macro"] == pytest.approx((1 + 0.4) / 2)
    assert m["top1_accuracy"] == 0.5 and m["top2_accuracy"] == 0.75


def test_missing_file_gives_no_rows(tmp_path):
    assert objf1.load_predictions_from_json(tmp_path / "missing.json") == []
