"""Core functions behind the paper numbers that had no direct test.

Feature loading and decomposition (step 5), concept text/image grounding, hook
helpers (step 4), concept-bank normalization, curve AUCs and the
insertion/deletion job list.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from tgcl.analysis.feature_decomposition import decompose_activations
from tgcl.analysis.feature_decomposition import get_feature_matrix
from tgcl.analysis.utils import cosine_similarity
from tgcl.analysis.utils import get_token_of_interest_features
import tgcl.helpers.utils as hutils

REPO = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------- features (step 5) --


def test_feature_matrix_stacks_per_sample_states():
    states = {"norm": [torch.full((1, 3), float(i)) for i in range(4)]}
    mat = get_feature_matrix(states, module_name="norm")
    assert mat.shape == (4, 3) and torch.equal(mat[:, 0], torch.arange(4.0))


def test_token_of_interest_filter_uses_one_flag_per_sample_across_batches():
    feats = torch.arange(10.0).reshape(5, 2)
    masks = [
        torch.tensor([True, False, True]),
        torch.tensor([False, True]),
    ]  # two batches
    kept = get_token_of_interest_features(feats, masks)
    assert torch.equal(kept, feats[[0, 2, 4]])
    assert torch.equal(get_token_of_interest_features(feats, None), feats)


def test_cosine_similarity_matrix():
    a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    sim = cosine_similarity(a, a)
    assert sim.shape == (2, 2)
    assert sim[0, 0] == pytest.approx(1.0) and sim[0, 1] == pytest.approx(
        2**-0.5, rel=1e-5
    )


def _args(**kw):
    base = dict(dl_alpha=0.1, dl_max_iter=200, seed=0)
    base.update(kw)
    return argparse.Namespace(**base)


def _two_cluster_data():
    rng = np.random.default_rng(0)
    a, b = np.zeros(6), np.zeros(6)
    a[:3], b[3:] = 1.0, 1.0
    x = np.concatenate(
        [
            a + 0.01 * rng.standard_normal((20, 6)),
            2 * b + 0.01 * rng.standard_normal((20, 6)),
        ]
    )
    return torch.tensor(x, dtype=torch.float32)


def test_snmf_codes_are_non_negative_and_separate_two_directions():
    x = _two_cluster_data()
    concepts, codes, _ = decompose_activations(
        x, num_concepts=2, decomposition_method="snmf", args=_args()
    )
    assert concepts.shape == (2, 6) and codes.shape == (40, 2)
    assert (codes >= 0).all()
    first, second = codes[:20].argmax(1), codes[20:].argmax(1)
    assert (
        len(set(first)) == 1 and len(set(second)) == 1 and first[0] != second[0]
    )


def test_pca_matches_sklearn():
    from sklearn.decomposition import PCA

    x = _two_cluster_data()
    concepts, codes, _ = decompose_activations(
        x, num_concepts=2, decomposition_method="pca", args=_args()
    )
    ref = PCA(n_components=2).fit(x.numpy())
    np.testing.assert_allclose(
        np.abs(concepts), np.abs(ref.components_), atol=1e-5
    )
    assert codes.shape == (40, 2)


def test_simple_takes_the_largest_norm_samples_as_concepts():
    x = torch.tensor([[1.0, 0.0], [5.0, 0.0], [0.0, 3.0], [0.1, 0.1]])
    concepts, codes, _ = decompose_activations(
        x, num_concepts=2, decomposition_method="simple", args=_args()
    )
    assert {tuple(c) for c in np.asarray(concepts)} == {(5.0, 0.0), (0.0, 3.0)}
    assert codes.max() == pytest.approx(
        1.0
    )  # 1 / (1 + distance 0) for a concept's own sample


# ------------------------------------------------------------ grounding ------


class _Tokenizer:

    def __init__(self, vocab):
        self.vocab = vocab

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self.vocab[i[0]] for i in ids]


def test_text_grounding_keeps_the_highest_logit_words_in_order():
    from tgcl.analysis.multimodal_grounding import concept_text_grounding

    vocab = ["the", "zebra", "apple", "car", "dog", "house", "tree", "river"]
    head = torch.nn.Linear(len(vocab), len(vocab), bias=False)
    head.weight.data = torch.eye(len(vocab))
    # logits = the concept itself: ranking river > tree > house > dog > car >
    # apple > zebra > "the"
    concept = torch.tensor([[0.0, 1, 2, 3, 4, 5, 6, 7]])
    words = concept_text_grounding(
        concept,
        lm_head=head,
        tokenizer=_Tokenizer(vocab),
        num_top_tokens=3,
        pre_num_top_tokens=8,
    )
    assert words == [
        ["river", "tree", "house"]
    ]  # not a hash-ordered subset; stop word "the" dropped


def test_remove_substrings_keeps_order_and_drops_contained_words():
    from tgcl.analysis.multimodal_grounding import remove_substrings

    assert remove_substrings(["cat", "dog", "cats", "dog", "bird"]) == [
        "dog",
        "cats",
        "bird",
    ]


def test_image_grounding_assigns_each_sample_to_one_concept():
    from tgcl.analysis.multimodal_grounding import concept_image_grounding

    acts = torch.tensor(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.0], [0.0, 0.0], [0.1, 0.3]]
    )
    idx = concept_image_grounding(acts, num_images_per_concept=0)
    assert idx[0].tolist() == [0, 2]  # strongest first
    assert idx[1].tolist() == [1, 4]
    assert (
        3 not in idx[0].tolist() + idx[1].tolist()
    )  # all-zero sample is unexplained
    assert [
        t.tolist()
        for t in concept_image_grounding(acts, num_images_per_concept=1)
    ] == [[0], [1]]


# ---------------------------------------------------------- hooks (step 4) ---


def test_hook_selects_the_active_altup_stream_of_gemma3n_layers():
    class AltUpLayer(torch.nn.Module):
        config = argparse.Namespace(altup_active_idx=1)

        def forward(self, x):
            return torch.stack(
                [x, x + 100, x + 200, x + 300]
            )  # (altup, B, T, D)

    layer = AltUpLayer()
    hutils.clear_hooks_variables()
    handle = layer.register_forward_hook(hutils.save_hidden_states("layer"))
    layer(torch.zeros(2, 3, 4))
    handle.remove()
    saved = hutils.HIDDEN_STATES["layer"][0]
    hutils.clear_hooks_variables()
    assert saved.shape == (2, 3, 4) and torch.all(saved == 100)


def test_update_dict_of_list_extends_batches_and_nested_dicts():
    acc = {}
    hutils.update_dict_of_list(
        {
            "image": ["a", "b"],
            "hidden_states": {"m": [torch.ones(1)]},
            "tag": "cat",
        },
        acc,
    )
    hutils.update_dict_of_list(
        {
            "image": ["c"],
            "hidden_states": {"m": [torch.zeros(1)]},
            "tag": "cat",
        },
        acc,
    )
    assert acc["image"] == ["a", "b", "c"] and acc["tag"] == ["cat", "cat"]
    assert [float(t) for t in acc["hidden_states"]["m"]] == [1.0, 0.0]


def test_batch_lengths_and_module_matching():
    assert hutils.get_batch_lengths([torch.zeros(3), [1, 2]]) == [3, 2]
    assert hutils.fmatch(
        "model.language_model.norm",
        ["model.language_model.norm"],
        exact_match=True,
    )
    assert not hutils.fmatch(
        "model.language_model.norm.weight",
        ["model.language_model.norm"],
        exact_match=True,
    )
    assert hutils.fmatch("model.layers.12.mlp", ["layers.*.mlp"])


def test_setup_hooks_rejects_hooks_without_modules():
    with pytest.raises(ValueError, match="modules_to_hook"):
        hutils.setup_hooks(
            torch.nn.Identity(),
            modules_to_hook=[["a"]],
            hook_names=["h1", "h2"],
        )


# ------------------------------------------------------- concept bank --------


def test_clean_example_ratio_env(monkeypatch):
    from tgcl.combine_concepts import parse_clean_example_ratio

    monkeypatch.setenv("CLEAN_EXAMPLE_RATIO", "0.6")
    assert parse_clean_example_ratio() == pytest.approx(0.6)
    for bad in ("1.5", "abc"):
        monkeypatch.setenv("CLEAN_EXAMPLE_RATIO", bad)
        assert parse_clean_example_ratio(default=0.8) == pytest.approx(0.8)


def test_gl_normalization_gives_unit_rows_and_rejects_other_methods():
    from tgcl.combine_concepts import apply_normalization

    out = apply_normalization(
        torch.tensor([[3.0, 4.0, 0.005], [0.0, -2.0, -0.004], [1.0, 1.0, 1.0]]),
        "gl",
    )
    assert out.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    for method in ("l2", "nope"):
        with pytest.raises(ValueError):
            apply_normalization(np.ones((2, 2)), method)


def test_unknown_decomposition_method_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown decomposition_method 'nope'"):
        decompose_activations(np.ones((4, 3), dtype=np.float32), 2, "nope")
