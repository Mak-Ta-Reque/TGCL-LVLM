"""Editable intermediates (scripts/intermediates.py) and their use by
paper_report."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pr():
    spec = importlib.util.spec_from_file_location(
        "paper_report", REPO / "scripts" / "paper_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run(tmp_path):
    """A pipeline run with every raw output the intermediates are made from."""
    run = tmp_path / "tgcl_seed1"
    bank_dir = run / "concept" / "snmf"
    bank_dir.mkdir(parents=True)
    torch.save(
        {
            "concepts": torch.tensor(
                [[1.0, 0.5, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0]]
            )
        },  # cosine 0.8
        bank_dir / "combined_concept_snmf_cr0.6_raw.pth",
    )
    (run / "features").mkdir()
    torch.save(
        {
            "concept": ["cat"] * 3,
            "token_of_interest_mask": [
                torch.tensor([True, False]),
                torch.tensor([True]),
            ],
        },
        run
        / "features"
        / "save_hidden_states_for_token_of_interest_results_cat.pth",
    )
    exp = run / "explanations" / "snmf"
    exp.mkdir(parents=True)
    results = [
        {
            "image_path": f"/val/{c}/{i}.jpg",
            "per_token_concepts": [
                {
                    "top_concepts": [
                        {"rank": 1, "concept_name": [p]},
                        {"rank": 2, "concept_name": ["dog"]},
                    ]
                }
            ],
        }
        for i, (c, p) in enumerate(
            [("cat", "cat"), ("cat", "dog"), ("dog", "dog")]
        )
    ]
    (exp / "vlm_explanations.json").write_text(json.dumps({"results": results}))
    ev = run / "eval" / "snmf"
    ev.mkdir(parents=True)
    with open(
        ev / "clip_bert_topk_per_sample.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample",
                "image_path",
                "rank_k",
                "bert",
                "clip",
                "random_bert",
                "random_clip",
            ]
        )
        w.writerows(
            [[0, "a", 1, 0.9, 0.6, 0.5, 0.4], [1, "b", 1, 0.7, 0.4, 0.5, 0.2]]
        )
    (ev / "concept_curves.json").write_text(
        json.dumps({"n_images": 3, "n_concepts": 2, "gap": 0.5})
    )
    return run


def test_export_writes_every_intermediate(pr, run):
    import tgcl.reporting.intermediates as im

    written = im.export_run(run, ["snmf"], [""])
    names = {p.relative_to(run / "intermediate").as_posix() for p in written}
    assert names == {
        "token_hits.csv",
        "snmf/concepts.npy",
        "snmf/object_predictions.csv",
        "snmf/retrieval.csv",
        "snmf/concept_curves.json",
    }
    assert (run / "intermediate" / "README.md").exists()
    assert im.read_concept_curves(run, "snmf")["gap"] == pytest.approx(0.5)
    assert im.read_token_hits(run) == [
        ("cat", True),
        ("cat", False),
        ("cat", True),
    ]
    assert im.read_object_predictions(run, "snmf") == [
        ("cat", "cat", "dog"),
        ("cat", "dog", "dog"),
        ("dog", "dog", "dog"),
    ]
    assert im.read_retrieval(run, "snmf")[1] == pytest.approx(
        {"bert": 0.8, "clip": 0.5, "random_bert": 0.5, "random_clip": 0.3}
    )
    assert np.load(run / "intermediate" / "snmf" / "concepts.npy").shape == (
        2,
        4,
    )


def test_metrics_follow_edited_intermediates_and_edits_survive(pr, run):
    import tgcl.reporting.intermediates as im

    im.export_run(run, ["snmf"], [""])
    assert pr.object_f1([run], "snmf")["top1_accuracy"] == pytest.approx(2 / 3)

    # The user fixes a label in the intermediate file.
    path = run / "intermediate" / "snmf" / "object_predictions.csv"
    path.write_text(
        path.read_text().replace(
            "/val/cat/1.jpg,cat,dog,dog", "/val/cat/1.jpg,cat,cat,dog"
        )
    )
    assert pr.object_f1([run], "snmf")["top1_accuracy"] == pytest.approx(1.0)

    # Re-export (as tables/ablation do) keeps the edit ...
    assert im.export_run(run, ["snmf"], [""]) == []
    assert pr.object_f1([run], "snmf")["top1_accuracy"] == pytest.approx(1.0)
    # ... unless explicitly overwritten.
    im.export_run(run, ["snmf"], [""], overwrite=True)
    assert pr.object_f1([run], "snmf")["top1_accuracy"] == pytest.approx(2 / 3)


def test_edited_concept_matrix_changes_quality(pr, run):
    import tgcl.reporting.intermediates as im

    im.export_run(run, ["snmf"], [""])
    before = pr.concept_quality([run], "snmf")["overlap"]
    np.save(
        run / "intermediate" / "snmf" / "concepts.npy",
        np.eye(2, 4, dtype=np.float32),
    )
    after = pr.concept_quality([run], "snmf")["overlap"]
    assert before == pytest.approx(0.8) and after == pytest.approx(0.0)


def test_word_alignment_intermediate_and_table(pr, run):
    import tgcl.reporting.intermediates as im

    ev = run / "eval" / "snmf"
    with open(
        ev / "word_alignment_per_sample.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample",
                "image_path",
                "top_concepts",
                "word_clipscore_top1",
                "word_clipscore_top2",
                "random_word_clipscore_top1",
                "random_word_clipscore_top2",
            ]
        )
        w.writerows(
            [
                [0, "a", "0 1", 0.8, 0.4, 0.5, 0.5],
                [1, "b", "1 0", 0.6, "", 0.3, 0.1],
            ]
        )
    im.export_run(run, ["snmf"], [""])
    got = im.read_word_alignment(run, "snmf")
    assert got["word_clipscore"] == {
        1: pytest.approx(0.7),
        2: pytest.approx((0.6 + 0.6) / 2),
    }  # empty cell skipped
    rows = pr.word_alignment_table([run], "snmf", eval_tags=("",))
    assert rows[0]["rank_k"] == 1 and rows[0][
        "word_clipscore"
    ] == pytest.approx(0.7)
    assert (
        rows[0]["random_word_clipscore"] == pytest.approx(0.4)
        and rows[0]["phrase_bertscore"] is None
    )


def test_scorer_writes_per_sample_rows(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "clip_bert_score_eval",
        REPO / "tgcl" / "evaluation" / "clip_bert_score_eval.py",
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    ranked = {1: {"bert": [0.9, 0.7], "clip": [0.6, 0.4]}}
    random_ = {1: {"bert": [0.5, 0.5], "clip": [0.4, 0.2]}}
    path = scorer.write_per_sample_scores(
        tmp_path / "p.csv",
        [{"image_path": "a"}, {"image_path": "b"}],
        ranked,
        random_,
    )
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [(r["sample"], r["image_path"], r["bert"]) for r in rows] == [
        ("0", "a", "0.9"),
        ("1", "b", "0.7"),
    ]
    assert scorer._summarize(ranked)[1]["bert_mean"] == pytest.approx(0.8)
