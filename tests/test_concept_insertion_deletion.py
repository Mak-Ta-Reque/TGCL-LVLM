"""tgcl/evaluation/concept_insertion_deletion.py with a toy output head."""

import json

import numpy as np
import pytest
import torch

from tgcl.evaluation import concept_insertion_deletion as cid


def test_concept_grid():
    assert cid.concept_grid(3).tolist() == [0, 1, 2, 3]
    g = cid.concept_grid(1000)
    assert g[0] == 0 and g[-1] == 1000 and list(g[:11]) == list(range(11))
    assert np.all(np.diff(g) > 0)


def test_ordered_states_endpoints_and_nesting():
    rng = np.random.default_rng(0)
    c = torch.tensor(rng.normal(size=(4, 9)))
    h = torch.tensor(rng.normal(size=9))
    order = torch.tensor([2, 0, 3, 1])
    deletion, insertion = cid.ordered_states(h, c, order, [0, 1, 2, 4])
    free = deletion[-1]  # h without the whole bank's span
    assert torch.allclose(deletion[0], h) and torch.allclose(insertion[-1], h)
    assert torch.allclose(insertion[0], free)
    assert torch.allclose(c @ free, torch.zeros(4, dtype=c.dtype), atol=1e-10)
    # after m steps the first m concepts of the order are gone from h
    assert torch.allclose(
        c[order[:2]] @ deletion[2], torch.zeros(2, dtype=c.dtype), atol=1e-10
    )


def test_curve_gap_signs():
    x = np.linspace(0, 1, 11)
    flat = np.ones_like(x)
    assert cid.curve_gap(x, flat, flat) == pytest.approx(0.0)
    # insertion rises at once, deletion falls at once: the maximum
    ins = np.where(x > 0, 1.0, 0.0)
    dele = np.where(x < 1e-9, 1.0, 0.0)
    assert cid.curve_gap(x, ins, dele) == pytest.approx(0.9, abs=1e-9)


def _toy(n_per_class=3, offset=0.0):
    """Answer states along their class axis (+ a shared offset); the toy head
    reads the class token's logit from that axis."""
    rng = np.random.default_rng(1)
    d, classes = 6, 3
    acts, results = [], []
    for k in range(classes):
        for i in range(n_per_class):
            s = np.zeros(d)
            s[k] = 4.0
            acts.append(s + offset + 0.01 * rng.normal(size=d))
            results.append(
                {
                    "image_path": f"/t/c{k}/{i}.jpg",
                    "per_token_concepts": [
                        {"token_id": k, "activation_index": len(acts) - 1}
                    ],
                }
            )
    w = torch.zeros(classes + 1, d, dtype=torch.float64)
    w[:classes, :classes] = torch.eye(classes)

    def token_probs(states, token_id):
        logits = states.double() @ w.T
        return torch.softmax(logits, dim=-1)[:, token_id]

    return {"results": results}, np.array(acts), token_probs


def test_class_concepts_carry_the_prediction():
    payload, acts, probs = _toy()
    concepts = torch.eye(6, dtype=torch.float64)[:3]
    result = cid.evaluate_payload(payload, acts, concepts, probs, seed=0)
    s = result["summary"]
    assert s["n_images"] == 9 and s["n_concepts"] == 3
    assert s["gap"] > 0.3 and s["gap"] > s["gap_random"]
    assert s["gap_net"] == pytest.approx(s["gap"] - s["gap_random"])
    curves = result["curves"]
    assert curves["del_value"].shape == (9, len(cid.concept_grid(3)))
    assert np.allclose(curves["del_value"][:, 0], 1.0)
    assert np.allclose(curves["ins_value"][:, -1], 1.0)


def test_unrelated_concepts_give_no_gap():
    payload, acts, probs = _toy()
    concepts = torch.eye(6, dtype=torch.float64)[3:]  # orthogonal to classes
    s = cid.evaluate_payload(payload, acts, concepts, probs)["summary"]
    assert s["gap"] == pytest.approx(0.0, abs=1e-6)


def test_centred_ranking_uses_the_calibration_mean():
    # A shared offset concept outranks the class axes in plain cosine; the
    # explainer's centred ranking puts the class axis first.
    payload, acts, probs = _toy(offset=3.0)
    concepts = torch.cat([torch.full((1, 6), 1.0), torch.eye(6)[:3]]).double()
    center = np.full(6, 3.0) + np.array([4, 4, 4, 0, 0, 0]) / 3
    plain = cid.evaluate_payload(payload, acts, concepts, probs)["summary"]
    centred = cid.evaluate_payload(payload, acts, concepts, probs, center)
    assert centred["summary"]["centred"] and not plain["centred"]
    assert centred["summary"]["gap"] > plain["gap"]


def test_write_and_read(tmp_path):
    payload, acts, probs = _toy()
    result = cid.evaluate_payload(
        payload, acts, torch.eye(6, dtype=torch.float64)[:3], probs
    )
    path = cid.write_outputs(result, tmp_path / "eval")
    assert json.loads(path.read_text()) == cid.read_summary(tmp_path / "eval")
    stored = np.load(tmp_path / "eval" / cid.CURVES)
    assert stored["ins_value"].shape == result["curves"]["ins_value"].shape
    assert cid.read_summary(tmp_path / "missing") is None
