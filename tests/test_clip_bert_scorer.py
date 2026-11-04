"""CLIP/BERT top-k scores (eval/clip_bert_score_eval.py) with deterministic fake
models."""

import csv
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def sc(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "clip_bert_score_eval",
        REPO / "tgcl" / "evaluation" / "clip_bert_score_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBERT:
    """F1 = 1 for identical strings, 0.5 otherwise; records every call."""

    def __init__(self):
        self.calls = []

    def score(self, cands, refs, verbose=False, batch_size=64):
        self.calls.append(list(zip(cands, refs)))
        return (
            None,
            None,
            torch.tensor([1.0 if c == r else 0.5 for c, r in zip(cands, refs)]),
        )


# Unit "embeddings": colour images and words map to fixed directions.
DIRS = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [-1.0, 0.0, 0.0],
}


@pytest.fixture
def fakes(sc, monkeypatch, tmp_path):
    bert = FakeBERT()
    monkeypatch.setattr(sc, "_bert_scorer", lambda device: bert)
    monkeypatch.setattr(
        sc, "_load_clip_once", lambda device="cpu": (None, None)
    )
    encoded = []

    def image_features(images, model, device, preprocess):
        encoded.append(len(images))
        return np.array(
            [
                DIRS[
                    {
                        (255, 0, 0): "red",
                        (0, 255, 0): "green",
                        (0, 0, 255): "blue",
                    }[img.getpixel((0, 0))]
                ]
                for img in images
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(sc, "_extract_image_features", image_features)
    monkeypatch.setattr(
        sc,
        "_extract_text_features",
        lambda texts, model, device: np.array(
            [DIRS.get(t, [0.0, 0.0, 1.0]) for t in texts]
        ),
    )
    paths = {}
    for name, rgb in (
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
    ):
        paths[name] = str(tmp_path / f"{name}.png")
        Image.new("RGB", (8, 8), rgb).save(paths[name])
    return {"bert": bert, "encoded": encoded, "paths": paths}


def _concept(name, image, rank):
    return {
        "rank": rank,
        "concept_name": [name],
        "image_grounding_path": [image],
        "image_grounding_bboxes": [None],
    }


def test_word_map_handles_subwords_and_multiword_answers(sc):
    assert sc.build_gt_token_map(
        ["garbage", "truck"], ["garbage", " truck"]
    ) == ["garbage", "truck"]
    assert sc.build_gt_token_map(["garbage"], ["gar", "bage"]) == [
        "garbage",
        "garbage",
    ]
    assert sc._split_text("Snowmobile, wild boar.") == [
        "snowmobile",
        "wild",
        "boar",
    ]


def test_extraction_drops_ignored_and_empty_words(sc):
    result = {
        "model_output": "No cup",
        "per_token_concepts": [
            {"token_text": "No", "top_concepts": [_concept("bowl", "x", 1)]},
            {
                "token_text": " cup",
                "top_concepts": [
                    _concept("cup", "a", 1),
                    _concept("bowl", "b", 2),
                ],
            },
        ],
    }
    rec = sc._extract_topk_predictions(result, (1, 2), frozenset({"no", "yes"}))
    assert rec["gt_token_word_map"] == ["cup"]
    assert rec["predictions_text"] == [{1: ["cup"], 2: ["bowl", "cup"]}]
    assert rec["prediction_image"] == [{1: ["a"], 2: ["a", "b"]}]
    empty = sc._extract_topk_predictions(
        {
            "model_output": "",
            "per_token_concepts": [
                {"token_text": "x", "top_concepts": [_concept("cup", "a", 1)]}
            ],
        },
        (1,),
    )
    assert empty["gt_token_word_map"] == []


def test_bert_averages_pairs_and_scores_each_unique_pair_once(sc, fakes):
    rec = {
        "gt_token_word_map": ["cat", "dog"],
        "predictions_text": [
            {1: ["cat"], 2: ["cat", "dog"]},
            {1: ["cat"], 2: ["cat"]},
        ],
        "prediction_image": [{1: [], 2: []}] * 2,
        "prediction_bboxes": [{1: [], 2: []}] * 2,
    }
    scores = sc._per_sample_scores([rec, rec, rec], (1, 2), "cpu", "cpu")
    # k=1: cat|cat=1, cat|dog=0.5 -> 0.75 ; k=2: word cat mean(1, 0.5)=0.75,
    # word dog 0.5 -> 0.625
    assert scores[1]["bert"] == [0.75] * 3 and scores[2]["bert"] == [0.625] * 3
    assert len(fakes["bert"].calls) == 1  # one batched call
    assert sorted(fakes["bert"].calls[0]) == sorted(
        {("cat", "cat"), ("dog", "cat"), ("cat", "dog")}
    )


def test_clip_score_formula_and_crop_cache(sc, fakes):
    p = fakes["paths"]
    rec = {
        "gt_token_word_map": ["red", "green"],
        "predictions_text": [{1: []}, {1: []}],
        "prediction_image": [{1: [p["red"], p["blue"]]}, {1: [p["red"]]}],
        "prediction_bboxes": [{1: [None, None]}, {1: [None]}],
    }
    scores = sc._per_sample_scores([rec, rec], (1,), "cpu", "cpu")
    # word red: crops red (cos 1 -> 2.5) and blue (cos -1 -> clipped 0) -> 1.25
    # ; word green vs red: cos 0 -> 0
    assert scores[1]["clip"] == [pytest.approx(0.625)] * 2
    assert sum(fakes["encoded"]) == 2  # red and blue crops encoded once each


def test_missing_grounding_image_raises(sc, fakes):
    rec = {
        "gt_token_word_map": ["red"],
        "predictions_text": [{1: []}],
        "prediction_image": [{1: ["/nonexistent/crop.png"]}],
        "prediction_bboxes": [{1: [None]}],
    }
    with pytest.raises(FileNotFoundError, match="grounding image not found"):
        sc._per_sample_scores([rec], (1,), "cpu", "cpu")


def test_unscoreable_samples_are_not_averaged(sc, fakes):
    empty = {
        "gt_token_word_map": [],
        "predictions_text": [],
        "prediction_image": [],
        "prediction_bboxes": [],
    }
    full = {
        "gt_token_word_map": ["cat"],
        "predictions_text": [{1: ["cat"]}],
        "prediction_image": [{1: []}],
        "prediction_bboxes": [{1: []}],
    }
    scores = sc._per_sample_scores([empty, full], (1,), "cpu", "cpu")
    assert scores[1]["bert"] == [None, 1.0]
    assert sc._summarize(scores)[1]["bert_mean"] == 1.0


def test_random_baseline_is_seeded(sc):
    bank = [
        {
            "concept_name": [f"c{i}"],
            "image_grounding_paths": [f"p{i}"],
            "image_grounding_bboxes": [None],
        }
        for i in range(5)
    ]
    a = sc._build_random_predictions(
        ["w1", "w2"], bank, (1, 2), random.Random(7)
    )
    b = sc._build_random_predictions(
        ["w1", "w2"], bank, (1, 2), random.Random(7)
    )
    assert a == b and all(len(t[2]) == 2 for t in a["predictions_text"])


def test_main_end_to_end_writes_table_and_per_sample_rows(
    sc, fakes, tmp_path, monkeypatch
):
    p = fakes["paths"]
    results = [
        {
            "image_path": "/val/red/1.JPEG",
            "model_output": "red",
            "per_token_concepts": [
                {
                    "token_text": "red",
                    "top_concepts": [
                        _concept("red", p["red"], 1),
                        _concept("green", p["green"], 2),
                    ],
                }
            ],
        },
        {
            "image_path": "/val/green/1.JPEG",
            "model_output": "no",
            "per_token_concepts": [
                {
                    "token_text": "no",
                    "top_concepts": [_concept("red", p["red"], 1)],
                }
            ],
        },
    ]
    (tmp_path / "exp.json").write_text(json.dumps({"results": results}))
    torch.save(
        {
            "concepts": torch.ones(2, 3),
            "concept_names": [["red"], ["green"]],
            "image_grounding_paths": [[p["red"]], [p["green"]]],
            "image_grounding_bboxes": [[None], [None]],
        },
        tmp_path / "bank.pth",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--json_path",
            str(tmp_path / "exp.json"),
            "--concept_path",
            str(tmp_path / "bank.pth"),
            "--max_k",
            "2",
            "--out_dir",
            str(tmp_path / "out"),
            "--ignore_words",
            "no,yes",
            "--bert_device",
            "cpu",
            "--clip_device",
            "cpu",
        ],
    )
    sc.main()
    rows = list(
        csv.DictReader(
            open(
                tmp_path / "out" / "clip_bert_topk_table.csv", encoding="utf-8"
            )
        )
    )
    assert [r["rank_k"] for r in rows] == ["1", "2"]
    assert float(rows[0]["bert_mean"]) == 1.0 and float(
        rows[0]["clip_mean"]
    ) == pytest.approx(2.5)
    per_sample = list(
        csv.DictReader(
            open(
                tmp_path / "out" / "clip_bert_topk_per_sample.csv",
                encoding="utf-8",
            )
        )
    )
    assert (
        len(per_sample) == 4 and per_sample[1]["bert"] == ""
    )  # the "no"-only answer is unscoreable
