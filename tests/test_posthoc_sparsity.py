"""Post-hoc concept sparsification and re-ranking stored explanations with a new
bank."""

import numpy as np
import pytest
import rerank_explanations as rx
import sparsify_concepts as sp
import torch


def _data(seed=0, n=400, d=40, rank=6):
    """Low-rank hidden states (as real ones) and two unit concepts in their
    span."""
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n, rank)) @ rng.standard_normal((rank, d))
    concepts = (
        rng.standard_normal((2, n)) @ states
    )  # dense directions inside the data's span
    return states, sp.unit(concepts)


@pytest.mark.parametrize("algorithm", ["omp", "magnitude"])
def test_sparsified_concepts_respect_the_budget(algorithm):
    states, concepts = _data()
    out = (
        sp.omp_sparsify(concepts, states, 8)
        if algorithm == "omp"
        else sp.magnitude_sparsify(concepts, 8)
    )
    assert out.shape == concepts.shape
    assert ((np.abs(out) > 0).sum(1) <= 8).all()
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def _pair_sims(c):
    return (c @ c.T)[~np.eye(len(c), dtype=bool)]


def test_centering_removes_the_shared_component():
    rng = np.random.default_rng(0)
    shared = rng.standard_normal(32)
    bank = sp.unit(rng.standard_normal((5, 32)) * 0.3 + shared)
    sims = _pair_sims
    centred = sp.center_concepts(bank)
    assert sims(bank).mean() > 0.8
    assert np.abs(sims(centred)).mean() < 0.4
    np.testing.assert_allclose(np.linalg.norm(centred, axis=1), 1.0)


def test_partial_centering_and_lowdin():
    rng = np.random.default_rng(1)
    bank = sp.unit(rng.standard_normal((5, 32)) * 0.3 + rng.standard_normal(32))
    half = np.abs(_pair_sims(sp.center_concepts(bank, 0.5))).mean()
    full = np.abs(_pair_sims(sp.center_concepts(bank, 1.0))).mean()
    assert full < half < np.abs(_pair_sims(bank)).mean()
    ortho = sp.lowdin_orthogonalize(bank)
    np.testing.assert_allclose(ortho @ ortho.T, np.eye(5), atol=1e-8)
    assert (np.diag(ortho @ bank.T) > 0.5).all()  # each stays near its own


def test_omp_keeps_the_scores_better_than_truncation():
    states, concepts = _data()
    omp = sp.fidelity(concepts, sp.omp_sparsify(concepts, states, 8), states)
    mag = sp.fidelity(concepts, sp.magnitude_sparsify(concepts, 8), states)
    assert omp["score_corr"] > mag["score_corr"] and omp["score_corr"] > 0.95


def test_rerank_matches_cosine_top_n_and_pooled_filter():
    bank = {
        "concepts": torch.eye(3, 4),
        "concept_names": [["a"], ["b"], ["c"]],
        "text_grounding": [["a"], ["b"], ["c"]],
        "image_grounding_paths": [[], [], []],
        "image_grounding_bboxes": [[], [], []],
    }
    acts = np.array(
        [[0.1, 2.0, 0.0, 0.0], [5.0, 0.0, 0.1, 0.0]]  # token "dog" -> concept 1
    )  # token "the" -> concept 0, ignored when pooling
    payload = {
        "results": [
            {
                "per_token_concepts": [
                    {
                        "token_text": "dog",
                        "activation_index": 0,
                        "top_concepts": [],
                    },
                    {
                        "token_text": "the",
                        "activation_index": 1,
                        "top_concepts": [],
                    },
                ],
                "top_concepts_over_sequence": [],
            }
        ]
    }
    out = rx.rerank(payload, acts, bank, top_n=2)
    toks = out["results"][0]["per_token_concepts"]
    assert [c["concept_index"] for c in toks[0]["top_concepts"]] == [1, 0]
    assert [c["concept_index"] for c in toks[1]["top_concepts"]] == [0, 2]
    assert (
        out["results"][0]["top_concepts_over_sequence"][0]["concept_index"] == 1
    )  # "the" not pooled
    assert set(out["concept_table"]) == {"0", "1", "2"} and out[
        "concept_table"
    ]["1"]["concept_name"] == ["b"]


def test_centred_rerank_ranks_like_the_explainer():
    from tgcl.inference import vlm_explainer_multibatch as ex

    rng = np.random.default_rng(0)
    concepts = torch.from_numpy(rng.normal(size=(6, 8)) + 3.0).float()
    acts = rng.normal(size=(5, 8)) + 3.0  # a shared component, as in the runs
    center = acts.mean(axis=0)
    payload = {
        "results": [
            {
                "per_token_concepts": [
                    {"token_text": f"w{i}", "activation_index": i}
                    for i in range(5)
                ],
                "top_concepts_over_sequence": [],
            }
        ]
    }
    out = rx.rerank(
        payload, acts, {"concepts": concepts}, top_n=6, center=center
    )
    explainer = object.__new__(ex.VLMConceptExplainer)
    explainer.concept_vectors = concepts
    explainer.set_centering(torch.from_numpy(center))
    sims = ex.cosine_to_concepts(
        explainer._centred(torch.from_numpy(acts)), explainer._unit_concepts
    )
    expected = torch.argsort(-sims, dim=1, stable=True).tolist()
    got = [
        [c["concept_index"] for c in t["top_concepts"]]
        for t in out["results"][0]["per_token_concepts"]
    ]
    assert got == expected
