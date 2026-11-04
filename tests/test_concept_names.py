"""Concept-name singularization (preprocessing/concept_names.py)."""

from types import SimpleNamespace

import pytest

from tgcl.preprocessing.concept_names import singularize
from tgcl.preprocessing.concept_names import singularize_phrase


@pytest.mark.parametrize(
    "word",
    [
        "gas",
        "bus",
        "grass",
        "glass",
        "lens",
        "cactus",
        "canvas",
        "dress",
        "species",
        "news",
        "chess",
        "fish",
        "brown",
        "visible",
        "was",
        "has",
    ],
)
def test_singular_words_unchanged(word):
    assert singularize(word) == word


@pytest.mark.parametrize(
    "plural,singular",
    [
        ("dogs", "dog"),
        ("buses", "bus"),
        ("gases", "gas"),
        ("trucks", "truck"),
        ("tables", "table"),
        ("bowls", "bowl"),
        ("benches", "bench"),
        ("knives", "knife"),
        ("mice", "mouse"),
        ("teeth", "tooth"),
        ("children", "child"),
        ("men", "man"),
        ("handbags", "handbag"),
    ],
)
def test_plurals_singularized(plural, singular):
    assert singularize(plural) == singular


def test_phrases():
    assert singularize_phrase("gas station") == "gas station"
    assert singularize_phrase("artificial grass") == "artificial grass"
    assert singularize_phrase("dining tables") == "dining table"
    assert singularize_phrase("") == ""


def test_select_top_concepts_normalization():
    from tgcl.preprocessing.select_top_concepts import _normalize_concept_name

    assert _normalize_concept_name("  Gas Station ") == "gas station"
    assert _normalize_concept_name("Garbage Trucks") == "garbage truck"


def test_gemma_shared_kv_wrapper_uses_throwaway_cache():
    import torch

    from tgcl.models.gemma3 import share_kv_without_cache

    calls = []

    class FakeTextModel(torch.nn.Module):

        def forward(self, input_ids=None, past_key_values=None, use_cache=None):
            calls.append(use_cache)
            return SimpleNamespace(
                last_hidden_state=input_ids,
                past_key_values="cache" if use_cache else None,
            )

    model = FakeTextModel()
    share_kv_without_cache(model)
    out = model(input_ids=torch.ones(1, 3), use_cache=False)
    assert (
        calls == [True] and out.past_key_values is None
    )  # cache built internally, then dropped
    out = model(
        input_ids=torch.ones(1, 3), past_key_values="given", use_cache=True
    )
    assert (
        calls[-1] is True and out.past_key_values == "cache"
    )  # normal cached generation untouched
