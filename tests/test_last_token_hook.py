"""save_hidden_states_given_token_idx, token_idx=-1 (MCD / CRAFT feature)."""

import argparse

import torch
from torch import nn

import tgcl.helpers.utils as utils


def test_last_input_token_of_left_padded_batch():
    model = nn.Sequential()
    model.add_module("layer", nn.Identity())
    args = argparse.Namespace(token_idx=-1, exact_match_modules_to_hook=True)
    utils.clear_hooks_variables()
    extract = utils.register_hooks(
        model,
        modules_to_hook=["layer"],
        hook_name="save_hidden_states_given_token_idx",
        args=args,
    )

    # Two left-padded rows of length 5, hidden size 3; row b, position i holds
    # 10*b + i.
    batch = torch.stack(
        [
            torch.arange(5, dtype=torch.float32)[:, None].repeat(1, 3) + 10 * b
            for b in range(2)
        ]
    )
    model(batch)  # one forward pass, as generate(max_new_tokens=1) does
    out = extract()
    utils.clear_hooks_variables()

    states = out["hidden_states"]["layer"]
    assert len(states) == 2
    assert torch.equal(states[0], torch.full((1, 3), 4.0))
    assert torch.equal(states[1], torch.full((1, 3), 14.0))
