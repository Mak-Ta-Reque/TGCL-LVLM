"""Concept-word alignment (eval/word_alignment_eval.py,
eval/word_alignment_phrases.py) on small inputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cx():
    return _load(
        "word_alignment_eval", "tgcl/evaluation/word_alignment_eval.py"
    )


@pytest.fixture(scope="module")
def ph():
    return _load(
        "word_alignment_phrases", "tgcl/evaluation/word_alignment_phrases.py"
    )


CORPUS = {"dog", "white", "people", "grass", "on", "cat"}
STOP = {"people"}


def test_word_filter_matches_the_original(cx):
    def valid(w):
        return cx.paper_valid_word(w, CORPUS, STOP)

    assert (
        valid(" Dog ")
        and valid("2")
        and not valid("on")
        and not valid("people")
        and not valid("dogz")
    )
    words = cx.concept_words(
        ["Dog", "dog", "on", "people", "white", "grass"] + ["cat"] * 3, valid
    )
    assert words == ["dog", "white", "grass", "cat"]
    assert (
        len(cx.concept_words([str(i) for i in range(20)], valid)) == 10
    )  # at most 10 words
    assert cx.caption_for(["dog", "white"]) == "dog, white."


def test_snmf_activation_equals_dictionary_learning_transform(cx):
    from sklearn.decomposition import DictionaryLearning

    rng = np.random.default_rng(0)
    concepts, x = rng.normal(size=(4, 6)), rng.normal(size=(5, 6))
    dl = DictionaryLearning(
        n_components=4,
        positive_code=True,
        transform_algorithm="lasso_cd",
        alpha=0.3,
    )
    dl.components_ = concepts
    assert np.allclose(
        cx.activations(x, concepts, "snmf", 0.3), dl.transform(x), atol=1e-6
    )
    assert np.allclose(cx.activations(x, concepts, "pca", 0), x @ concepts.T)
    km = cx.activations(x, concepts, "simple", 0)
    assert km.shape == (5, 4) and np.all((km > 0) & (km <= 1))
    with pytest.raises(ValueError, match="sae2"):
        cx.activations(x, concepts, "sae2", 0)


def test_random_words_are_seeded_filtered_and_sized(cx):
    vocab = ["dog", "on", "white", "grass", "cat", "xx", "people", "the"]
    weight = torch.randn(len(vocab), 8)

    def valid(w):
        return cx.paper_valid_word(w, CORPUS, STOP)

    a = cx.random_words(weight, lambda t: vocab[t], [2, 1], valid, seed=3)
    assert a == cx.random_words(
        weight, lambda t: vocab[t], [2, 1], valid, seed=3
    )
    assert [len(w) <= n for w, n in zip(a, [2, 1])] == [True, True]
    assert all(valid(w) for ws in a for w in ws)


def test_random_words_reach_the_wanted_count_past_junk_tokens(cx):
    # 40 junk tokens rank above the valid words for every random direction: a
    # cut at 10 n tokens
    # (the original) left the concept without random words, so its random
    # baseline was NaN
    vocab = [f"##{i}" for i in range(40)] + ["dog", "grass", "people", "cat"]
    weight = torch.cat([torch.full((40, 4), 10.0), torch.ones(4, 4)])

    def valid(w):
        return cx.paper_valid_word(w, CORPUS, STOP)

    words = cx.random_words(weight, lambda t: vocab[t], [3, 0], valid, seed=0)
    assert (
        len(words[0]) == 3
        and len(set(words[0])) == 3
        and all(valid(w) for w in words[0])
    )
    assert words[1] == []


def test_first_scored_token_is_the_token_of_interest(cx):
    results = [
        {
            "image_path": "a.jpg",
            "per_token_concepts": [
                {"token_text": "No", "activation_index": 0},
                {"token_text": " dog", "activation_index": 1},
            ],
        },
        {
            "image_path": "b.jpg",
            "per_token_concepts": [{"token_text": ",", "activation_index": 2}],
        },
    ]
    assert cx.first_token_rows(results, frozenset({"no"})) == [(0, "a.jpg", 1)]


def test_clip_scores_formula(cx):
    img = np.array([[1.0, 0.0], [0.0, 1.0]])
    caps = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    top = np.array([[0, 2], [1, 0]])
    assert np.allclose(
        cx.clip_scores(img, caps, top), [[2.5, 0.0], [2.5, 0.0]]
    )  # negative cosine clipped


class FakeBERT:

    def __init__(self):
        self.calls = 0

    def score(self, cands, refs, verbose=False, batch_size=256):
        self.calls += 1
        return (
            None,
            None,
            torch.tensor(
                [
                    len(set(c.split()) & set(r.split())) / 4
                    for c, r in zip(cands, refs)
                ]
            ),
        )


def test_bert_takes_the_best_phrase_caption_pair(cx):
    phrases = [["a dog runs", "white dog runs"], ["a cat sleeps"]]
    captions = [["a dog on grass", "white dog runs fast"], []]
    scorer = FakeBERT()
    out = cx.bert_concept_scores(
        phrases, captions, np.array([[0, 1], [1, 0]]), scorer
    )
    assert scorer.calls == 1
    assert out[0, 0] == pytest.approx(
        0.75
    )  # best pair: "white dog runs" vs "white dog runs fast" = 3/4
    assert out[0, 1] == pytest.approx(
        0.25
    )  # "a cat sleeps" vs "a dog on grass": a
    assert np.isnan(out[1]).all()  # no captions -> no score


def test_merged_words_bertscore_scores_one_text_per_concept(cx):
    words = [["dog", "grass"], []]  # second concept has no grounded words
    assert cx.merged_words(words) == [["dog, grass."], []]
    captions = [["dog, on grass.", "a cat"]]
    out = cx.bert_concept_scores(
        cx.merged_words(words), captions, np.array([[0, 1]]), FakeBERT()
    )
    assert out[0, 0] == pytest.approx(
        2 / 4
    )  # "dog," and "grass." shared with the first caption
    assert np.isnan(out[0, 1])  # no words -> no score


def test_topk_summary_skips_unscored(cx):
    s = np.array([[1.0, 0.0, 0.5], [np.nan, np.nan, np.nan], [0.5, 0.5, 0.5]])
    summary = cx.topk_summary(s, (1, 3))
    assert summary[1] == (pytest.approx(0.75), pytest.approx(0.25), 2)
    assert summary[3][0] == pytest.approx(0.5) and summary[3][2] == 2


def test_phrase_prompt_and_parsing(ph):
    assert ph.prompt_for(["dog", "white"]).startswith(
        "Generate 3 distinct phrases, each incorporating the words dog, white."
    )
    answer = (
        'Here you go:\n1. "A white dog plays."\n2) The dog is white.\n3 - '
        "White dogs run.\n4. extra"
    )
    assert ph.parse_phrases(answer) == [
        "A white dog plays.",
        "The dog is white.",
        "White dogs run.",
    ]
    calls = []
    known = ph.build_phrases(
        [["dog"], ["dog"], []],
        lambda p: calls.append(p) or "1. a\n2. b\n3. c",
        {},
    )
    assert (
        known == {"dog": ["a", "b", "c"]} and len(calls) == 1
    )  # cached, empty set skipped


def test_coco_caption_map(tmp_path):
    cap = _load("build_coco_captions", "scripts/build_coco_captions.py")
    (tmp_path / "captions_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "000000000007.jpg"}],
                "annotations": [
                    {"image_id": 7, "caption": "A dog. "},
                    {"image_id": 7, "caption": "Two dogs"},
                ],
            }
        )
    )
    assert cap.build([tmp_path / "captions_val2017.json"]) == {
        "000000000007.jpg": ["A dog.", "Two dogs"]
    }
