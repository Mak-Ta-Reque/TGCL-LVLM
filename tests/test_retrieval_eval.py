"""Concept-retrieval evaluation pieces: run layout (EVAL_TAG), explainer
text-only input, and the scorer's --ignore_words."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest

from tgcl.helpers import run_layout

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- run layout


def test_untagged_layout_is_the_historical_one(tmp_path):
    assert run_layout.explanations_dir(tmp_path) == tmp_path / "explanations"
    assert run_layout.eval_dir(tmp_path, "") == tmp_path / "eval"
    assert run_layout.plots_dir(tmp_path) == tmp_path / "plots"
    assert run_layout.explanations_json(tmp_path, "snmf") == (
        tmp_path / "explanations" / "snmf" / "vlm_explanations.json"
    )


def test_tagged_layout(tmp_path):
    assert (
        run_layout.explanations_dir(tmp_path, "text_only")
        == tmp_path / "explanations_text_only"
    )
    assert run_layout.eval_dir(tmp_path, " grid ") == tmp_path / "eval_grid"


@pytest.mark.parametrize("bad", ["../x", "a/b", "-x", "a b"])
def test_unsafe_tags_are_rejected(bad):
    with pytest.raises(ValueError):
        run_layout.validate_eval_tag(bad)


def test_find_concept_bank(tmp_path):
    bank_dir = tmp_path / "concept" / "snmf"
    bank_dir.mkdir(parents=True)
    assert run_layout.find_concept_bank(tmp_path, "snmf") is None
    (bank_dir / "combined_negative_concept_snmf_cr0.6_raw.pth").touch()
    bank = bank_dir / "combined_concept_snmf_cr0.6_raw.pth"
    bank.touch()
    assert run_layout.find_concept_bank(tmp_path, "snmf") == bank
    (bank_dir / "combined_concept_snmf_cr0_raw.pth").touch()
    with pytest.raises(ValueError, match="Several concept banks"):
        run_layout.find_concept_bank(tmp_path, "snmf")


# ---------------------------------------------------------------- explainer


@pytest.fixture(scope="module")
def explainer_module():
    from tgcl.inference import vlm_explainer_multibatch

    return vlm_explainer_multibatch


def test_text_only_instruction_uses_repeat_prompt(explainer_module):
    from tgcl.models.constants import TASK_PROMPTS

    prompt = TASK_PROMPTS[explainer_module.TEXT_ONLY_PROMPT_TEMPLATE][
        "Repeat the text"
    ]
    assert (
        explainer_module.text_only_instruction("gas_station")
        == f"{prompt}: 'gas station'"
    )
    with pytest.raises(ValueError):
        explainer_module.text_only_instruction(None)


def test_prompt_label_precedence(explainer_module):
    paths = ["/data/val/church/a.jpg", "/data/val/dog/b.jpg"]
    resolve = explainer_module.resolve_prompt_labels
    assert resolve(paths) == ["church", "dog"]
    assert resolve(paths, ground_truth_labels=["x", "y"]) == ["x", "y"]
    assert resolve(paths, ground_truth_labels=["x", "y"], prompt_label="z") == [
        "z",
        "z",
    ]
    assert resolve([None]) == [None]


def test_explainer_rejects_unknown_input_mode(explainer_module):
    with pytest.raises(ValueError, match="input_mode"):
        explainer_module.VLMConceptExplainer(
            "model", "bank.pth", "layer", input_mode="audio"
        )


# ---------------------------------------------------------------- scorer


@pytest.fixture(scope="module")
def scorer():
    """eval/clip_bert_score_eval.py with its scoring backends stubbed (only the
    token/concept extraction is under test)."""
    sys.modules.setdefault("bert_score", types.SimpleNamespace(score=None))
    sys.modules.setdefault("clip", types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location(
        "clip_bert_score_eval",
        REPO / "tgcl" / "evaluation" / "clip_bert_score_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(answer, tokens):
    return {
        "model_output": answer,
        "per_token_concepts": [
            {
                "token_text": tok,
                "top_concepts": [{"concept_name": [f"c_{tok.strip()}"]}],
            }
            for tok in tokens
        ],
    }


def test_ignore_words_drop_no_yes_tokens(scorer):
    result = _result("No apple", ["No", " apple"])
    kept = scorer._extract_topk_predictions(
        result, (1,), frozenset({"no", "yes"})
    )
    assert kept["gt_token_word_map"] == ["apple"]
    assert kept["predictions_text"] == [{1: ["c_apple"]}]


def test_no_ignore_words_keeps_every_token(scorer):
    result = _result("No apple", ["No", " apple"])
    default = scorer._extract_topk_predictions(result, (1,))
    assert default["gt_token_word_map"] == [
        "no",
        "apple",
    ]  # answer words are lower-cased
    assert len(default["predictions_text"]) == 2
