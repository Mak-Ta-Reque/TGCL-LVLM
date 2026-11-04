"""run_full_pipeline._downstream_outputs: what a recomputed step invalidates."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location(
        "run_full_pipeline", REPO / "scripts" / "run_full_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_full_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, eval_tag: str):
    run = tmp_path / "run"
    for d in (
        "concept",
        "explanations",
        "explanations_grid",
        "eval",
        "eval_grid",
        "plots",
        "plots_grid",
        "intermediate/snmf",
        "features/features",
        "inference",
    ):
        (run / d).mkdir(parents=True, exist_ok=True)
    for f in (
        "object_predictions.csv",
        "object_predictions_grid.csv",
        "word_alignment.csv",
        "word_alignment_grid.csv",
        "concept_curves_grid.json",
        "concepts.npy",
    ):
        (run / "intermediate" / "snmf" / f).touch()

    def tagged(name):
        return run / (f"{name}_{eval_tag}" if eval_tag else name)

    return SimpleNamespace(
        output_dir=run,
        crops_json=run / "inference" / "crops.json",
        logitlens_dir=run / "logitlens",
        features_dir=run,
        decomp_dir=run / "concept",
        eval_tag=eval_tag,
        explain_dir=tagged("explanations"),
        eval_dir=tagged("eval"),
        plots_dir=tagged("plots"),
    )


def test_new_concepts_invalidate_every_variant(rp, tmp_path):
    cfg = _run(tmp_path, "")
    out = {p for p, _ in rp._downstream_outputs(cfg, 5)}
    run = cfg.output_dir
    assert {
        run / "explanations",
        run / "explanations_grid",
        run / "eval_grid",
        run / "plots_grid",
        run / "intermediate",
    } <= out
    assert (
        run / "concept" not in out and run / "features" / "features" not in out
    )


def test_step7_rerun_invalidates_only_its_variant(rp, tmp_path):
    cfg = _run(tmp_path, "grid")
    out = {p for p, _ in rp._downstream_outputs(cfg, 7)}
    inter = cfg.output_dir / "intermediate" / "snmf"
    assert cfg.output_dir / "plots_grid" in out
    assert {
        inter / "word_alignment_grid.csv",
        inter / "concept_curves_grid.json",
        inter / "object_predictions_grid.csv",
    } <= out
    assert (
        inter / "word_alignment.csv" not in out
        and inter / "concepts.npy" not in out
    )  # base variant untouched
    assert cfg.output_dir / "eval_grid" not in out  # step 7's own output


def test_step6_rerun_also_drops_its_evaluation(rp, tmp_path):
    cfg = _run(tmp_path, "")
    out = {p for p, _ in rp._downstream_outputs(cfg, 6)}
    assert (
        cfg.output_dir / "eval" in out
        and cfg.output_dir / "eval_grid" not in out
    )
    assert (
        cfg.output_dir / "intermediate" / "snmf" / "word_alignment.csv" in out
    )
