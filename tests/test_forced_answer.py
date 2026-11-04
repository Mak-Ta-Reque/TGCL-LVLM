"""Step 4's forced-answer features (FORCED_RESPONSE): the state AT the class
word's tokens."""

import torch

from tgcl.helpers import utils
import tgcl.save_features as sf

CONCEPT = (7, 8)  # a two-token class word


class _Model:
    """Records the forward call; the test simulates the hidden-state hook."""

    def __init__(self):
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, logits_to_keep=0):
        self.calls.append(
            dict(input_ids=input_ids, logits_to_keep=logits_to_keep)
        )

    __call__ = forward


class _ModelClass:

    def preprocessor(self, instruction, image_file, response, generation_mode):
        assert not generation_mode and response == "There is a car."
        ids = (
            [1, 2, 3] if image_file == "short" else [1, 2, 3, 4, 4]
        )  # different prompt lengths
        ids = (
            ids + [5, 6] + list(CONCEPT) + [9]
        )  # "There is a" + class word + end
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }


def test_forced_fields_put_the_state_at_the_class_word():
    model = _Model()
    fields = sf.forced_answer_fields(
        model,
        _ModelClass(),
        ["short", "long"],
        "Describe the image.",
        "There is a car.",
        pad_id=0,
    )
    assert model.calls[0]["logits_to_keep"] == 1  # no full-vocabulary logits
    ids = torch.stack(fields["model_output"])
    assert ids.shape == (2, 10) and ids[0, 0] == 0  # left-padded batch
    # hidden state of position t = t (row 0) and 100 + t (row 1), as one forward
    # pass records it
    utils.clear_hooks_variables()
    utils.HIDDEN_STATES["layer"] = [
        torch.stack([torch.arange(10.0), 100 + torch.arange(10.0)])[:, :, None]
    ]
    out = utils.get_hidden_states(
        extract_token_of_interest=True,
        token_sequences=[CONCEPT],
        save_only_generated_tokens=True,
        image=["short", "long"],
        **fields,
    )
    states = torch.stack(
        [s.reshape(-1) for s in out["hidden_states"]["layer"]]
    ).reshape(-1)
    # class word at positions 7 and 8 in both rows: the mean of the states AT
    # them, not one before
    assert states.tolist() == [7.5, 107.5]
    assert out["token_of_interest_mask"].tolist() == [True, True]
    utils.clear_hooks_variables()


def test_hooks_pause_records_nothing():
    utils.clear_hooks_variables()
    hook = utils.save_hidden_states("layer")
    with utils.hidden_state_hooks_paused():
        hook(None, None, torch.ones(1, 2, 3))
    assert "layer" not in utils.HIDDEN_STATES
    hook(None, None, torch.ones(1, 2, 3))
    assert len(utils.HIDDEN_STATES["layer"]) == 1
    utils.clear_hooks_variables()


def test_explainer_states_at_tokens_matches_forced_alignment():
    from tgcl.inference import vlm_explainer_multibatch as ex

    # prompt of L=4 positions, 3 generated tokens ("gold", "fish", <eos>):
    # captured positions 0..L+T-2 = 0..5
    captured = torch.arange(6.0)[:, None]
    predicting = captured[-3:].reshape(-1).tolist()
    at = ex.states_at_tokens(captured, 3).reshape(-1).tolist()
    assert predicting == [
        3.0,
        4.0,
        5.0,
    ]  # position L-1 predicts "gold", L predicts "fish", ...
    assert at == [
        4.0,
        5.0,
        5.0,
    ]  # "gold" sits at L=4, "fish" at 5; <eos> has no own state
    assert ex.states_at_tokens(captured, 1).reshape(-1).tolist() == [5.0]
