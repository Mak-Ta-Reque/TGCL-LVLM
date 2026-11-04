"""Concept-span token-of-interest extraction (src/helpers/utils.py)."""

import argparse

import pytest
import torch
from torch import nn

from tgcl.helpers.utils import encode_concept_token_sequences
from tgcl.helpers.utils import extract_token_of_interest_states
from tgcl.helpers.utils import find_concept_span
import tgcl.helpers.utils as utils


class PieceTokenizer:
    """SentencePiece-like toy tokenizer.

    A word after a space becomes the piece "▁word"; a word at the start of the
    text has no marker. Words listed in `splits` break into several pieces (only
    the first carries the marker). With `stray_space=True`, a leading space is
    emitted as its own "▁" piece, as some real tokenizers do.
    """

    def __init__(self, splits=None, stray_space=False):
        self.splits = splits or {}
        self.stray_space = stray_space
        self.vocab = {}
        self.inverse = {}

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 10
            self.inverse[self.vocab[piece]] = piece
        return self.vocab[piece]

    def encode(self, text, add_special_tokens=False):
        ids = []
        if self.stray_space and text.startswith(" "):
            ids.append(self._id("▁"))
            text = text.lstrip(" ")
        for position, word in enumerate(text.split(" ")):
            pieces = self.splits.get(word, [word])
            marked = position > 0 or text.startswith(" ")
            for k, piece in enumerate(pieces):
                ids.append(self._id(("▁" if marked and k == 0 else "") + piece))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.inverse[int(i)] for i in ids).replace("▁", " ")


def hidden(length, dim=4):
    """Row i is filled with the value i, so averages are easy to check."""
    return torch.arange(length, dtype=torch.float32)[:, None].repeat(1, dim)


# ---------------------------------------------------------------- encoding


def test_single_token_word_has_one_id_sequences():
    tok = PieceTokenizer()
    sequences = encode_concept_token_sequences("apple", tok)
    assert all(len(seq) == 1 for seq in sequences)
    pieces = {tok.inverse[seq[0]] for seq in sequences}
    assert {"apple", "▁apple", "Apple", "▁Apple", "apples", "▁apples"} <= pieces


def test_multi_token_word_has_full_sequence():
    tok = PieceTokenizer(splits={"handbag": ["hand", "bag"]})
    sequences = encode_concept_token_sequences("handbag", tok)
    assert (tok.vocab["▁hand"], tok.vocab["bag"]) in sequences
    assert (tok.vocab["hand"], tok.vocab["bag"]) in sequences


def test_underscore_tag_matches_spaced_words():
    tok = PieceTokenizer()
    sequences = encode_concept_token_sequences("dining_table", tok)
    assert sequences == encode_concept_token_sequences("dining table", tok)
    assert (tok.vocab["▁dining"], tok.vocab["▁table"]) in sequences


def test_stray_leading_space_piece_is_dropped():
    tok = PieceTokenizer(stray_space=True)
    sequences = encode_concept_token_sequences("cat", tok)
    space_id = tok.vocab["▁"]
    assert all(seq[0] != space_id for seq in sequences)


def test_empty_word_is_rejected():
    with pytest.raises(ValueError):
        encode_concept_token_sequences(" _ ", PieceTokenizer())


# ---------------------------------------------------------------- span search


def test_find_span_earliest_then_longest():
    assert find_concept_span([5, 1, 2, 3, 1, 2], [(1, 2), (1, 2, 3)]) == (1, 4)
    assert find_concept_span([1, 9, 1, 2], [(1, 2), (9,)]) == (1, 2)
    assert find_concept_span(torch.tensor([4, 4]), [(1,)]) is None
    assert find_concept_span([1, 2], [(1, 2, 3)]) is None


# ---------------------------------------------------------------- extraction


def test_no_answer_uses_concept_token_only():
    tok = PieceTokenizer()
    answer = torch.tensor([tok.encode("No apple")])  # ["No", "▁apple"]
    states, found = extract_token_of_interest_states(
        tokens=hidden(2)[None],
        pred_tokens=answer,
        token_sequences=encode_concept_token_sequences("apple", tok),
    )
    assert found.tolist() == [True]
    assert torch.equal(states[0, 0], hidden(2)[1])


def test_multi_token_concept_is_averaged():
    tok = PieceTokenizer(splits={"handbag": ["hand", "bag"]})
    answer = torch.tensor([tok.encode("No handbag here")])  # No ▁hand bag ▁here
    states, found = extract_token_of_interest_states(
        tokens=hidden(4)[None],
        pred_tokens=answer,
        token_sequences=encode_concept_token_sequences("handbag", tok),
    )
    assert found.tolist() == [True]
    assert torch.allclose(states[0, 0], hidden(4)[1:3].mean(dim=0))


def test_plural_span_covers_suffix_piece():
    tok = PieceTokenizer(splits={"apples": ["apple", "s"]})
    answer = torch.tensor([tok.encode("two apples")])  # two ▁apple s
    states, _ = extract_token_of_interest_states(
        tokens=hidden(3)[None],
        pred_tokens=answer,
        token_sequences=encode_concept_token_sequences("apple", tok),
    )
    assert torch.allclose(states[0, 0], hidden(3)[1:3].mean(dim=0))


def test_missing_concept_is_masked_out():
    tok = PieceTokenizer()
    states, found = extract_token_of_interest_states(
        tokens=hidden(2)[None],
        pred_tokens=torch.tensor([tok.encode("a dog")]),
        token_sequences=encode_concept_token_sequences("cat", tok),
    )
    assert found.tolist() == [False]
    assert torch.equal(states[0, 0], hidden(2)[0])


def test_per_sample_sequences_and_padding():
    tok = PieceTokenizer()
    pred = torch.tensor([tok.encode("No cat") + [-1], tok.encode("yes a dog")])
    states, found = extract_token_of_interest_states(
        tokens=torch.stack([hidden(3), hidden(3) + 100]),
        pred_tokens=pred,
        token_sequences_per_sample=[
            encode_concept_token_sequences("cat", tok),
            encode_concept_token_sequences("dog", tok),
        ],
    )
    assert found.tolist() == [True, True]
    assert torch.equal(states[0, 0], hidden(3)[1])
    assert torch.equal(states[1, 0], hidden(3)[2] + 100)


def test_legacy_single_id_input():
    states, found = extract_token_of_interest_states(
        tokens=hidden(3)[None],
        pred_tokens=torch.tensor([[7, 8, 9]]),
        token_of_interest_idx=8,
    )
    assert found.tolist() == [True]
    assert torch.equal(states[0, 0], hidden(3)[1])


def test_requires_some_concept_input():
    with pytest.raises(ValueError):
        extract_token_of_interest_states(
            tokens=hidden(2)[None], pred_tokens=torch.tensor([[1, 2]])
        )


# ---------------------------------------------------------------- hook path


def test_registered_hook_extracts_span_from_generated_tokens():
    """Drives register_hooks -> forward hook -> get_hidden_states exactly as
    feature extraction does with --save_only_generated_tokens."""
    tok = PieceTokenizer(splits={"handbag": ["hand", "bag"]})
    prompt = tok.encode("Classify the image")
    generated = tok.encode("No handbag")  # No ▁hand bag
    full = prompt + generated

    model = nn.Sequential()
    model.add_module("layer", nn.Identity())
    args = argparse.Namespace(
        token_of_interest="handbag",
        token_of_interest_idx=None,
        token_of_interest_start_token=0,
        save_only_generated_tokens=True,
        exact_match_modules_to_hook=True,
    )
    utils.clear_hooks_variables()
    extract = utils.register_hooks(
        model,
        modules_to_hook=["layer"],
        hook_name="save_hidden_states_for_token_of_interest",
        tokenizer=tok,
        args=args,
    )
    model(hidden(len(full))[None])
    out = extract(
        model_output=torch.tensor([full]),
        model_generated_output=[generated],
        image=["img.jpg"],
    )
    utils.clear_hooks_variables()

    states = out["hidden_states"]["layer"][0]
    expected = hidden(len(full))[len(prompt) + 1 : len(prompt) + 3].mean(dim=0)
    assert out["token_of_interest_mask"].tolist() == [True]
    assert torch.allclose(states[0], expected)
