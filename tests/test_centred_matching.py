"""Centred concept matching: the explainer's set_centering, MCQ choices for
labelled calibration images, and the pipeline's calibration list
(EXPL_CENTER=calibration)."""

import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ex():
    return _load(
        "vlm_explainer_multibatch",
        REPO / "tgcl" / "inference" / "vlm_explainer_multibatch.py",
    )


def test_centring_removes_the_hub(ex):
    # every concept and every answer state share a big component (+10 on
    # coordinate 0); classes differ on 1, 2
    concepts = torch.tensor([[10.0, 1, 0], [10.0, 0, 1]])
    answers = torch.tensor(
        [[10.0, 0.2, 0], [10.0, 0, 0.2]]
    )  # answer i belongs to concept i
    bare = object.__new__(ex.VLMConceptExplainer)
    bare.concept_vectors = concepts
    bare._answer_center = None
    bare._unit_concepts = torch.nn.functional.normalize(concepts, dim=1)
    raw = ex.cosine_to_concepts(
        answers + torch.tensor([0.0, 0.3, 0]), bare._unit_concepts
    )
    assert raw.argmax(1).tolist() == [0, 0]  # one concept wins both: a hub
    bare.set_centering(
        torch.tensor([10.0, 0.1, 0.1])
    )  # mean of calibration answers
    centred = ex.cosine_to_concepts(bare._centred(answers), bare._unit_concepts)
    assert centred.argmax(1).tolist() == [0, 1]


def test_mcq_choices_take_an_explicit_label(ex):
    pool = ["acorn", "toucan", "school bus", "goldfish"]
    choices = ex.sample_mcq_choices(
        "data/x/train_all/whatever.JPEG",
        pool,
        2,
        seed=0,
        true_classes=["toucan"],
    )
    assert "toucan" in choices and len(choices) == 3


@pytest.fixture(scope="module")
def rp():
    return _load("run_full_pipeline", REPO / "scripts" / "run_full_pipeline.py")


def _config(tmp_path: Path, single: bool):
    cmap = tmp_path / "map.json"
    json.dump(
        {
            "acorn": [f"a{i}.jpg" for i in range(8)],
            "toucan": [f"t{i}.jpg" for i in range(3)],
            "goldfish": [f"g{i}.jpg" for i in range(8)],
            "school bus": [f"s{i}.jpg" for i in range(8)],
        },
        open(cmap, "w", encoding="utf-8"),
    )
    return SimpleNamespace(
        seed=0,
        expl_calibration_per_class=5,
        input_dir=tmp_path / "train",
        explain_dir=tmp_path
        / ("explanations" if single else "explanations_grid"),
        single_object=single,
        _map=cmap,
    )


def test_calibration_list_single_and_grid(rp, tmp_path, monkeypatch):
    for single in (True, False):
        cfg = _config(tmp_path, single)
        monkeypatch.setattr(
            rp, "_active_concept_map", lambda config, logger: config._map
        )
        path = rp._calibration_list(cfg, logging.getLogger("t"))
        data = json.load(open(path, encoding="utf-8"))
        if single:
            assert (
                len(data["images"]) == 5 + 3 + 5 + 5 == len(data["labels"])
            )  # toucan has only 3
            for p, label in zip(data["images"], data["labels"]):
                assert (
                    Path(p).parent == cfg.input_dir
                    and Path(p).name[0] == label[0]
                )  # image of its class
            again = json.load(
                open(
                    rp._calibration_list(cfg, logging.getLogger("t")),
                    encoding="utf-8",
                )
            )
            assert again == data  # deterministic
        else:
            assert len(data["grids"]) == 4 * 5 // 4
            for grid in data["grids"]:
                assert len(set(grid["labels"])) == 4 and len(grid["cells"]) == 4


def test_label_starts_are_the_first_token_of_each_label(ex):
    assert ex.label_start_indices(["school", " bus"]) == [0]
    assert ex.label_start_indices(
        ["acorn", ",", " school", " bus", ",", " tou", "can"]
    ) == [0, 2, 5]
    assert ex.label_start_indices(["", " gold", "fish", "<|im_end|>"]) == [1]


def test_concept_fields_copy_bank_entries_and_fill_gaps_with_none(ex):
    bare = object.__new__(ex.VLMConceptExplainer)
    bare.concept_names = [["car"], ["dog"]]
    bare.text_grounding = [["wheel", "road"]]
    bare.image_grounding_paths = [("a.jpg", "b.jpg"), "c.jpg"]
    bare.image_grounding_bboxes = [[[0, 0, 4, 4]], 7]
    bare.image_grounding_masks = None

    assert bare._concept_fields(0) == {
        "concept_name": ["car"],
        "text_grounding": ["wheel", "road"],
        "image_grounding_path": ["a.jpg", "b.jpg"],
        "image_grounding_bboxes": [[0, 0, 4, 4]],
        "image_grounding_masks": None,
    }
    assert bare._concept_fields(1) == {
        "concept_name": ["dog"],
        "text_grounding": None,
        "image_grounding_path": "c.jpg",
        "image_grounding_bboxes": "7",
        "image_grounding_masks": None,
    }
