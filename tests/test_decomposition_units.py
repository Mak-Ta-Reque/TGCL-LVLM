"""Step 5 decompositions on small random features (CPU)."""

import argparse

import numpy as np
import pytest
import torch

from tgcl.analysis.feature_decomposition import decompose_activations
from tgcl.analysis.feature_decomposition import decompose_tag
from tgcl.helpers.sae import l2_normalize_rows
from tgcl.helpers.sae import SparseAutoencoder
from tgcl.helpers.sae import train_sae


def _args(method, **extra):
    return argparse.Namespace(
        decomposition_method=method,
        num_concepts=[3],
        dl_alpha=0.1,
        dl_max_iter=50,
        seed=0,
        **extra,
    )


def _features(n=40, d=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


@pytest.mark.parametrize("method", ["snmf", "pca", "simple"])
def test_shapes_of_every_method(method):
    components, activations, _ = decompose_activations(
        _features(), 3, method, args=_args(method)
    )
    assert np.asarray(components).shape == (3, 16)
    assert np.asarray(activations).shape == (40, 3)


def test_snmf_codes_are_non_negative():
    _, activations, _ = decompose_activations(
        _features(), 3, "snmf", args=_args("snmf")
    )
    assert (np.asarray(activations) >= 0).all()


def _hoyer(c):
    c = np.asarray(c, dtype=np.float64)
    n = c.shape[1]
    ratio = np.abs(c).sum(1) / np.linalg.norm(c, axis=1)
    return float(np.mean((np.sqrt(n) - ratio) / (np.sqrt(n) - 1)))


def test_snmf_dict_l1_makes_concepts_sparser():
    mat = np.abs(_features(n=60, d=32)) + 0.5  # a shared positive component
    dense, _, _ = decompose_activations(mat, 3, "snmf", args=_args("snmf"))
    sparse, codes, _ = decompose_activations(
        mat, 3, "snmf", args=_args("snmf", dl_dict_alpha=0.3)
    )
    assert _hoyer(sparse) > _hoyer(dense) + 0.1
    assert (np.asarray(sparse) == 0).any()  # exact zeros, not just small
    assert (np.abs(sparse).max(axis=1) > 0).all()  # no concept wiped out
    assert (np.linalg.norm(sparse, axis=1) <= 1 + 1e-5).all()
    assert (np.asarray(codes) >= 0).all() and codes.shape == (60, 3)


def test_snmf_dict_l1_off_by_default():
    a = decompose_activations(_features(), 3, "snmf", args=_args("snmf"))[0]
    b = decompose_activations(
        _features(), 3, "snmf", args=_args("snmf", dl_dict_alpha=0.0)
    )[0]
    np.testing.assert_array_equal(a, b)


def test_decompose_tag_is_reproducible_in_any_order():
    first = decompose_tag(_features(seed=1), _args("snmf"))[0]
    decompose_tag(_features(seed=2), _args("snmf"))  # consumes the RNGs
    again = decompose_tag(_features(seed=1), _args("snmf"))[0]
    np.testing.assert_array_equal(first, again)


def test_bfloat16_tensor_input_is_accepted():
    mat = torch.from_numpy(_features()).to(torch.bfloat16)
    components, _, _ = decompose_activations(mat, 3, "pca", args=_args("pca"))
    assert np.asarray(components).shape == (3, 16)


def test_sae_top_k_keeps_exactly_k_codes():
    sae = SparseAutoencoder(16, 8, top_k=2)
    _, z = sae(torch.from_numpy(_features()))
    assert ((z != 0).sum(dim=1) <= 2).all()


def test_sae_training_lowers_the_reconstruction_loss():
    torch.manual_seed(0)
    data = l2_normalize_rows(torch.from_numpy(_features(n=128)))
    sae = SparseAutoencoder(16, 8, sparsity_lambda=0.0)
    before = sae.loss(data, *sae(data)).item()
    train_sae(sae, data, epochs=30, batch_size=32, device="cpu")
    sae.eval()
    with torch.no_grad():
        after = sae.loss(data, *sae(data)).item()
    assert after < before
