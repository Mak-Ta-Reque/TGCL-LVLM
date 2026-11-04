"""Concept-quality metrics (tgcl/reporting/dictionary_metrics.py) and
scripts/dictionary_metrics.py on synthetic banks with known answers."""

import dictionary_metrics as script
import numpy as np
import pytest
import torch

from tgcl.reporting import dictionary_metrics as dm


def _orthonormal(k, d, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, k)))
    return q.T  # (k, d), orthonormal rows


# ------------------------------------------------------------------ sparsity


def test_gini_sparsity_one_hot_flat_and_scale():
    assert dm.gini_sparsity(np.eye(3, 8)) == pytest.approx(1 - 1 / 8)
    assert dm.gini_sparsity(np.ones((2, 8))) == pytest.approx(0.0)
    w = np.random.default_rng(1).normal(size=(4, 32))
    assert dm.gini_sparsity(w * 50) == pytest.approx(dm.gini_sparsity(w))
    trimmed = np.where(np.abs(w) >= np.quantile(np.abs(w), 0.7), w, 0.0)
    assert dm.gini_sparsity(trimmed) > dm.gini_sparsity(w)


# ------------------------------------------------------------------ overlap


def test_l2_overlap_orthogonal_identical_and_opposite():
    assert dm.l2_overlap(_orthonormal(5, 16)) == pytest.approx(0.0, abs=1e-12)
    same = np.ones((3, 8))
    assert dm.l2_overlap(same) == pytest.approx(1.0)
    assert dm.l2_overlap(np.stack([same[0], -same[0]])) == pytest.approx(1.0)
    assert np.isnan(dm.l2_overlap(np.ones((1, 8))))


def test_l2_overlap_is_scale_free_and_drops_tiny_coordinates():
    w = np.random.default_rng(2).normal(size=(4, 32))
    assert dm.l2_overlap(w * 300.0) == pytest.approx(dm.l2_overlap(w))
    a = np.array([[1.0, 0.0, 0.005], [0.0, 1.0, 0.005]])  # tiny shared coord
    assert dm.l2_overlap(a) == pytest.approx(0.0)
    assert dm.l2_overlap(a, threshold=0.0) > 0.0


# ---------------------------------------------------------------- stability


def test_instability_ignores_order_and_averages_seed_pairs():
    rng = np.random.default_rng(3)
    w = rng.normal(size=(5, 12))
    assert dm.instability([w, w[[3, 0, 4, 1, 2]]]) == pytest.approx(
        0.0, abs=1e-12
    )
    banks = [rng.normal(size=(4, 10)) for _ in range(3)]
    pairwise = [
        1 - dm.matched_cosine(banks[i], banks[j])
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    assert dm.instability(banks) == pytest.approx(np.mean(pairwise))
    assert np.isnan(dm.instability([w]))


# ------------------------------------------------------------------- script


def test_tgcl_combination_orthogonalizes_and_keeps_25_percent():
    w = dm.unit_rows(np.random.default_rng(4).normal(size=(4, 40)) + 1.0)
    assert np.array_equal(script.post_processing("none")(w), w)
    out = script.post_processing("lowdin+magnitude:0.25")(w)
    assert ((np.abs(out) > 0).sum(1) == 10).all()  # 0.25 * 40
    ortho = script.post_processing("lowdin")(w)
    assert dm.l2_overlap(ortho, threshold=0.0) < 1e-6
    with pytest.raises(ValueError):
        script.post_processing("nope")(w)


def _write_bank(path, concepts):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"concepts": torch.tensor(concepts, dtype=torch.float32)}, path)


def test_load_bank_counts_rejected_directions_only_when_asked(tmp_path):
    folder = tmp_path / "concept" / "snmf"
    _write_bank(folder / "combined_concept_snmf_cr0_raw.pth", np.eye(2, 6))
    _write_bank(
        folder / "combined_negative_concept_snmf_cr0_raw.pth", np.eye(6)[2:5]
    )
    assert script.load_bank(tmp_path, "snmf", False).shape == (2, 6)
    assert script.load_bank(tmp_path, "snmf", True).shape == (5, 6)
    assert script.load_bank(tmp_path / "missing", "snmf", True) is None


def test_evaluate_row_end_to_end(tmp_path, monkeypatch):
    w = _orthonormal(3, 12)
    runs = []
    for seed in (1, 2):
        run = tmp_path / f"run_seed{seed}"
        _write_bank(run / "concept/snmf/combined_concept_snmf_cr0.6_raw.pth", w)
        runs.append(str(run))
    monkeypatch.setattr(script, "ROOT", tmp_path.parent)
    row = {
        "dataset": "toy",
        "method": "m",
        "decomposition": "snmf",
        "runs": runs,
    }
    records = script.evaluate_row(row)
    assert [r["post_processing"] for r in records] == list(script.POSTS)
    rec = records[0]
    assert rec["status"] == "ok" and rec["n_concepts"] == "3/3"
    assert rec["gini_sparsity_mean"] == pytest.approx(dm.gini_sparsity(w))
    assert rec["l2_overlap_mean"] < 1e-3  # float32 storage + 1 % threshold
    assert rec["instability"] == pytest.approx(0.0, abs=1e-6)
    missing = script.evaluate_row(row | {"runs": [str(tmp_path / "x")]})
    assert {r["status"] for r in missing} == {"no bank"}
