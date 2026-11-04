"""DETERMINISTIC=1 turns on strict GPU determinism; else nothing changes."""

import os

import torch

from tgcl.helpers.utils import apply_determinism_setting


def _flags():
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def _restore(flags):
    enabled, det, bench, tf32_mm, tf32_cudnn = flags
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = bench
    torch.backends.cuda.matmul.allow_tf32 = tf32_mm
    torch.backends.cudnn.allow_tf32 = tf32_cudnn


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("DETERMINISTIC", raising=False)
    before = _flags()
    assert apply_determinism_setting() is False
    assert _flags() == before


def test_on_with_the_variable(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = _flags()
    try:
        assert apply_determinism_setting() is True
        assert _flags() == (True, True, False, False, False)
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        _restore(before)
