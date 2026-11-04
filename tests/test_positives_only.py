"""DECOMPOSE_POSITIVES_ONLY: step 5 learns from the segments answered with the
class word only."""

import argparse

import torch

from tgcl.analysis import load_features
from tgcl.analysis.utils import mask_to_positive_answers

PREDS = ["car", "No car", "car.", "no car", "Car"]


def test_mask_keeps_batches_and_drops_negatives():
    mask = [torch.tensor([True, True, True]), torch.tensor([True, False])]
    out = mask_to_positive_answers(mask, PREDS)
    assert [m.tolist() for m in out] == [[True, False, True], [False, False]]


def test_load_features_positives_only(tmp_path, monkeypatch):
    data = {
        "hidden_states": {
            "layer": [torch.full((1, 4), float(i)) for i in range(5)]
        },
        "token_of_interest_mask": [torch.tensor([True] * 5)],
        "model_predictions": PREDS,
        "image": [f"img{i}" for i in range(5)],
    }
    path = tmp_path / "save_hidden_states_for_token_of_interest_results_car.pth"
    torch.save(data, path)
    args = argparse.Namespace(
        load_matched_features=False,
        module_to_decompose="layer",
        decomposition_extract_pos=None,
    )
    monkeypatch.setenv("DECOMPOSE_POSITIVES_ONLY", "1")
    feats, meta = load_features(str(path), args=args)
    (feat,), (m,) = feats.values(), meta.values()
    assert feat[:, 0].tolist() == [0.0, 2.0, 4.0]  # the positive samples
    assert torch.cat(m["token_of_interest_mask"]).tolist() == [
        True,
        False,
        True,
        False,
        True,
    ]
    monkeypatch.setenv("DECOMPOSE_POSITIVES_ONLY", "0")
    feats, _ = load_features(str(path), args=args)
    assert next(iter(feats.values())).shape[0] == 5  # default: every sample
