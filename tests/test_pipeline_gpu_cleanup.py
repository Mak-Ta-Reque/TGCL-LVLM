"""run_full_pipeline._cleanup_gpu never creates a CUDA context in the pipeline's
parent process."""

import importlib.util
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parent.parent


def _pipeline():
    spec = importlib.util.spec_from_file_location(
        "run_full_pipeline", REPO / "scripts" / "run_full_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_full_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def test_cleanup_skips_cuda_when_not_initialized(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a: calls.append("synchronize")
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.append("empty_cache")
    )
    _pipeline()._cleanup_gpu()
    assert calls == []  # a context on GPU 0 would be created otherwise
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    _pipeline()._cleanup_gpu()
    assert calls == ["empty_cache", "synchronize"]
