"""tgcl.combine_features: pooled feature files stay compact and exact."""

import torch

from tgcl.combine_features import combine_pth_files
from tgcl.helpers import utils


def _feature_file(path, n, seq=300, dim=64, offset=0.0):
    """A step-4 file whose states are slices of full token sequences, as
    get_hidden_states produced them before it cloned."""
    full = torch.arange(n * seq * dim, dtype=torch.float32).reshape(n, seq, dim)
    full += offset
    states = [full[i : i + 1, -1:, :] for i in range(n)]  # views, no copies
    torch.save(
        {
            "hidden_states": {"model.language_model.norm": states},
            "image": [f"img_{i}" for i in range(n)],
            "selected_layer": "model.language_model.norm",
        },
        path,
    )
    return [s.clone() for s in states]


def test_combined_file_holds_only_the_states(tmp_path):
    a = _feature_file(tmp_path / "save_a.pth", 5)
    b = _feature_file(tmp_path / "save_b.pth", 4, offset=0.5)
    sliced_size = (tmp_path / "save_a.pth").stat().st_size
    combine_pth_files(str(tmp_path))
    out = tmp_path / "combined_features.pth"
    data = torch.load(out, map_location="cpu", weights_only=False)
    states = data["hidden_states"]["model.language_model.norm"]
    assert len(states) == 9
    for got, want in zip(states, a + b):
        assert torch.equal(got, want)
    assert data["image"][5] == "img_0"
    assert data["selected_layer"] == ["model.language_model.norm"] * 2
    # 9 states of 64 floats: a few KB, not the 300-token sequences behind them
    assert out.stat().st_size < sliced_size / 20


def test_token_idx_states_do_not_keep_the_sequence(monkeypatch):
    full = torch.randn(2, 300, 16)
    monkeypatch.setattr(utils, "HIDDEN_STATES", {"layer": [full]})
    out = utils.get_hidden_states(token_idx=-1)
    per_sample = out["hidden_states"]["layer"]  # one tensor per sample
    got = torch.cat([v.reshape(1, 16) for v in per_sample])
    assert torch.equal(got, full[:, -1, :])
    for v in per_sample:
        assert (
            v.untyped_storage().nbytes() < full.untyped_storage().nbytes() / 100
        )
