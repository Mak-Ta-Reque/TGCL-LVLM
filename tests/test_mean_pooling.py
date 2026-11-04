"""mean_skip_special_tokens (src/helpers/utils.py), the full-output pooling."""

import warnings

import pytest
import torch

from tgcl.helpers.utils import mean_skip_special_tokens

SPECIAL = {0, 1}


def hidden(rows):
    """(B, L, 1) hidden states from nested lists."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(-1)


def test_skips_special_tokens():
    out = mean_skip_special_tokens(
        hidden([[1.0, 2.0, 9.0]]), torch.tensor([[5, 6, 1]]), SPECIAL
    )
    assert out.shape == (1, 1, 1)
    assert out.item() == pytest.approx(1.5)


def test_padding_is_not_counted():
    out = mean_skip_special_tokens(
        hidden([[2.0, 4.0, 0.0, 0.0]]), torch.tensor([[5, 6, -1, -1]]), SPECIAL
    )
    assert out.item() == pytest.approx(3.0)


def test_all_special_falls_back_to_mean_of_real_tokens():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mean_skip_special_tokens(
            hidden([[4.0, 8.0, 0.0]]), torch.tensor([[0, 1, -1]]), SPECIAL
        )
    assert out.item() == pytest.approx(6.0)
    assert any("special" in str(w.message) for w in caught)


def test_empty_answer_gives_zeros():
    out = mean_skip_special_tokens(
        hidden([[3.0, 3.0]]), torch.tensor([[-1, -1]]), SPECIAL
    )
    assert out.item() == 0.0


def test_rows_are_independent():
    out = mean_skip_special_tokens(
        hidden([[1.0, 3.0], [5.0, 7.0]]),
        torch.tensor([[5, 6], [0, 6]]),
        SPECIAL,
    )
    assert out[:, 0, 0].tolist() == pytest.approx([2.0, 7.0])
