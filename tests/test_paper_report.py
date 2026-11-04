"""scripts/paper_report.py on small synthetic runs."""

import csv
import importlib.util
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


def _bank(run, concepts, method="snmf"):
    d = run / "concept" / method
    d.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"concepts": torch.as_tensor(concepts, dtype=torch.float32)},
        d / f"combined_concept_{method}_cr0.6_raw.pth",
    )


def _clip_bert(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank_k",
                "bert_mean",
                "clip_mean",
                "random_bert_mean",
                "random_clip_mean",
            ],
        )
        w.writeheader()
        w.writerows(rows)


ALL_TAGS = ["", "grid", "text_only", "image_only", "combined"]


def _export(pr, *runs):
    """Pipeline outputs -> intermediate files (what every metric reads)."""
    pr.export_intermediates(runs, ["snmf"], ALL_TAGS)


def test_group_runs_by_seed_suffix(pr, tmp_path):
    runs = [tmp_path / n for n in ("a_seed2", "a_seed1", "b", "a_seedling")]
    groups = pr.group_runs(runs)
    assert list(groups) == ["a", "a_seedling", "b"]
    assert [r.name for r in groups["a"]] == ["a_seed1", "a_seed2"]


def test_concept_quality_threshold_and_instability(pr, tmp_path):
    atoms = np.array([[1.0, 0.0005, 0.0, 0.2], [0.0, 1.0, 0.004, 0.0]])
    _bank(tmp_path / "x_seed1", atoms)
    _bank(tmp_path / "x_seed2", atoms[::-1].copy())  # same atoms, other order
    runs = [tmp_path / "x_seed1", tmp_path / "x_seed2"]
    _export(pr, *runs)

    q = pr.concept_quality(runs, "snmf", near_zero_threshold=1e-3)
    assert q["n_seeds"] == 2 and q["n_concepts"] == 2
    # |c_i| / ||c|| below 1e-3: atom 1 -> 0.0005, 0.0 ; atom 2 -> 0.0, 0.0
    assert q["near_zero_frac"] == pytest.approx((2 / 4 + 2 / 4) / 2)
    assert q["instability"] == pytest.approx(
        0.0, abs=1e-9
    )  # matching ignores order
    coarse = pr.concept_quality(runs, "snmf", near_zero_threshold=1e-2)
    assert coarse["near_zero_frac"] == pytest.approx((2 / 4 + 3 / 4) / 2)

    single = pr.concept_quality(runs[:1], "snmf")
    assert single["instability"] is None and single["overlap_std"] is None


def test_overlap_undefined_for_a_single_concept(pr, tmp_path):
    _bank(tmp_path / "one_seed1", np.array([[1.0, 0.5, 0.0, 0.0]]))
    _export(pr, tmp_path / "one_seed1")
    assert (
        pr.concept_quality([tmp_path / "one_seed1"], "snmf")["overlap"] is None
    )


def test_retrieval_columns(pr, tmp_path):
    run = tmp_path / "tgcl_seed1"
    values = {"text_only": 0.5, "image_only": 0.6, "combined": 0.7}
    for tag, v in values.items():
        _clip_bert(
            run / f"eval_{tag}" / "snmf" / "clip_bert_topk_table.csv",
            [
                {
                    "rank_k": 1,
                    "bert_mean": v,
                    "clip_mean": v / 2,
                    "random_bert_mean": v - 0.4,
                    "random_clip_mean": v / 4,
                }
            ],
        )
    _export(pr, run)
    rows = {r["metric"]: r for r in pr.retrieval_table([run], "snmf")}
    assert rows["bert"] == {
        "rank_k": 1,
        "metric": "bert",
        "Random": pytest.approx(0.2),
        "Text-only": 0.5,
        "Image-only": 0.6,
        "Combined": 0.7,
    }
    assert rows["clip"]["Random"] == pytest.approx(
        0.15
    )  # from the image-only run


def test_faithfulness_means_concept_curve_summaries_over_seeds(pr, tmp_path):
    import json

    runs = []
    for seed, gap in ((1, 0.6), (2, 0.8)):
        run = tmp_path / f"c_seed{seed}"
        out = run / "eval" / "snmf"
        out.mkdir(parents=True)
        (out / "concept_curves.json").write_text(
            json.dumps(
                {
                    "n_images": 10,
                    "n_concepts": 5,
                    "gap": gap,
                    "gap_random": 0.1,
                    "gap_net": gap - 0.1,
                    "insertion_auc": 0.9,
                    "deletion_auc": 0.2,
                }
            )
        )
        runs.append(run)
    _export(pr, *runs)
    row = pr.faithfulness(runs, "snmf")
    assert row["n_seeds"] == 2 and row["n_images"] == 10
    assert row["gap"] == pytest.approx(0.7)
    assert row["gap_net"] == pytest.approx(0.6)
    assert row["gap_std"] == pytest.approx(0.1)
    assert pr.faithfulness(runs, "snmf", "grid") is None


def test_markdown_bolds_best_per_direction(pr):
    rows = [
        {"name": "a", "up": 0.9, "down": 0.2},
        {"name": "b", "up": 0.5, "down": 0.1},
        {"name": "c", "up": None, "down": 0.3},
    ]
    md = pr.markdown_table(
        rows,
        [
            ("name", "Name", None, None),
            ("up", "Up", 2, "max"),
            ("down", "Down", 1, "min"),
        ],
    )
    lines = md.strip().splitlines()
    assert lines[2] == "| a | **0.90** | 0.2 |"
    assert lines[3] == "| b | 0.50 | **0.1** |"
    assert lines[4] == "| c | – | 0.3 |"


def test_near_zero_counts_coordinates_below_0_1_of_the_unit_concept(pr):
    assert pr.NEAR_ZERO_THRESHOLD == pytest.approx(0.1)
    c = torch.tensor([[0.6, 0.79, 0.05, 0.0]])  # ~unit norm: 2 of 4 below 0.1
    assert pr.near_zero_fraction(c) == pytest.approx(0.5)
    assert pr.near_zero_fraction(c * 50) == pytest.approx(
        0.5
    )  # scale does not matter


def test_overlap_uses_the_same_threshold(pr, tmp_path):
    # the atoms share only a small coordinate (0.05): it counts at 1e-3, is
    # dropped at the 0.1 default
    atoms = np.array([[1.0, 0.0, 0.05], [0.0, 1.0, 0.05]])
    _bank(tmp_path / "o_seed1", atoms)
    _export(pr, tmp_path / "o_seed1")
    assert pr.concept_quality([tmp_path / "o_seed1"], "snmf")[
        "overlap"
    ] == pytest.approx(0.0)
    fine = pr.concept_quality(
        [tmp_path / "o_seed1"], "snmf", near_zero_threshold=1e-3
    )
    assert fine["overlap"] == pytest.approx(0.0025 / 1.0025)


def test_gini_effective_coordinates_rank_and_score_correlation(pr):
    one_hot, flat = torch.eye(1, 8), torch.ones(1, 8)
    assert pr.gini_index(one_hot) == pytest.approx(
        1 - 1 / 8
    )  # maximum 1 - 1/n (0.9995 for 2048-d concepts)
    assert pr.gini_index(flat) == pytest.approx(0.0)
    assert pr.gini_index(one_hot * 5) == pytest.approx(
        pr.gini_index(one_hot)
    )  # scale-free
    assert pr.effective_coordinates(one_hot) == pytest.approx(1.0)
    assert pr.effective_coordinates(flat) == pytest.approx(8.0)
    assert pr.effective_rank_fraction(torch.eye(4, 8)) == pytest.approx(
        1.0
    )  # orthogonal
    assert pr.effective_rank_fraction(torch.ones(4, 8)) == pytest.approx(
        0.25
    )  # identical: 1/K
    rng = np.random.default_rng(0)
    states = rng.standard_normal((200, 8))
    c = torch.tensor(
        [
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [2.0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    corr = pr.score_correlation(c, states)  # pairs: 1, ~0, ~0
    assert corr == pytest.approx(1 / 3, abs=0.05)
    assert pr.score_correlation(c[:1], states) is None


def test_hoyer_is_one_for_one_hot_and_zero_for_flat(pr):
    one_hot, flat = torch.eye(2, 8), torch.ones(2, 8)
    assert pr.hoyer_selectivity(one_hot) == pytest.approx(1.0)
    assert pr.hoyer_selectivity(flat) == pytest.approx(0.0, abs=1e-6)
    assert pr.hoyer_selectivity(one_hot * 7) == pytest.approx(1.0)


def test_overlap_zero_for_orthogonal_one_for_identical(pr):
    assert pr.concept_overlap(torch.eye(3, 8)) == pytest.approx(0.0)
    same = torch.ones(3, 8) / 8**0.5
    assert pr.concept_overlap(same, 0.01) == pytest.approx(1.0)


def test_overlap_threshold_is_relative_to_the_l2_norm(pr):
    rng = np.random.default_rng(0)
    bank = torch.from_numpy(rng.normal(size=(5, 64)))
    unit = bank / bank.norm(dim=1, keepdim=True)
    assert pr.concept_overlap(bank * 200, 0.1) == pytest.approx(
        pr.concept_overlap(unit, 0.1)
    )


def test_stability_matching_ignores_order_and_pca_signs(pr):
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(4, 16))
    shuffled = bank[[2, 0, 3, 1]]
    assert pr.matched_cosine_similarity(bank, shuffled) == pytest.approx(1.0)
    flipped = shuffled * np.array([[1], [-1], [1], [-1]])
    assert pr.matched_cosine_similarity(bank, flipped) < 1.0
    assert pr.matched_cosine_similarity(
        bank, flipped, sign_invariant=True
    ) == pytest.approx(1.0)
