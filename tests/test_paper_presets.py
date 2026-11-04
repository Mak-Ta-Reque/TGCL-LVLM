"""configs/paper/*.env: every dataset x method x evaluation preset is a valid
pipeline config, sourced exactly as the README does."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import pytest
from test_pipeline_config import PIPELINE_ENV_VARS

REPO = Path(__file__).resolve().parent.parent
PRESETS = REPO / "configs" / "paper"

DATASETS = ["coco", "imagenet"]
METHODS = ["tgcl_sam", "tgcl_sliding_window", "mcd", "craft", "coxlmm"]
EVALS = [
    None,
    "eval_grid",
    "eval_text_only",
    "eval_image_only",
    "eval_combined",
]
EXTRA_VARS = [
    "IMAGE_ROOT_SINGLE",
    "IMAGE_ROOT_GRID",
    "MASK_FULL_IMAGE",
    "INPAINTING_METHOD",
]


@pytest.fixture(scope="module")
def pipeline():
    sys.modules.setdefault(
        "dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
    )
    spec = importlib.util.spec_from_file_location(
        "run_full_pipeline", REPO / "scripts" / "run_full_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sourced_env(*files: str) -> dict:
    """Variables after `set -a; . f1; . f2 ...` in a clean bash."""
    script = (
        "set -a; "
        + " ".join(f'. "{PRESETS / (f + ".env")}";' for f in files)
        + " env -0"
    )
    out = subprocess.run(
        ["env", "-i", "bash", "-c", script], capture_output=True, check=True
    ).stdout
    return dict(
        item.split("=", 1) for item in out.decode().split("\0") if "=" in item
    )


def make_config(pipeline, monkeypatch, tmp_path, env: dict):
    monkeypatch.setattr(pipeline, "ROOT_DIR", tmp_path)
    for key in PIPELINE_ENV_VARS + EXTRA_VARS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if (
            key in PIPELINE_ENV_VARS
            or key in EXTRA_VARS
            or key in ("TOKEN_IDX", "SLIDING_WINDOW_TAG_FILTER")
        ):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "run"))
    return pipeline.PipelineConfig()


@pytest.mark.parametrize("eval_file", EVALS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_preset_is_valid(
    pipeline, monkeypatch, tmp_path, dataset, method, eval_file
):
    files = ["common", dataset, method] + ([eval_file] if eval_file else [])
    env = sourced_env(*files)
    config = make_config(pipeline, monkeypatch, tmp_path, env)

    base = make_config(
        pipeline, monkeypatch, tmp_path, sourced_env("common", dataset, method)
    )
    # Evaluation variants must find the base run's concepts.
    assert (config.decomp_dir, config.ratio_tag) == (
        base.decomp_dir,
        base.ratio_tag,
    )
    assert env["IMAGE_ROOT"] == (
        env["IMAGE_ROOT_GRID"]
        if eval_file == "eval_grid"
        else env["IMAGE_ROOT_SINGLE"]
    )
    assert (
        config.eval_tag
        == {
            None: "",
            "eval_grid": "grid",
            "eval_text_only": "text_only",
            "eval_image_only": "image_only",
            "eval_combined": "combined",
        }[eval_file]
    )


@pytest.mark.parametrize(
    "method, hook, tokens, strategy, crop",
    [
        (
            "tgcl_sam",
            "save_hidden_states_for_token_of_interest",
            50,
            "per_tag",
            "langsam",
        ),
        (
            "tgcl_sliding_window",
            "save_hidden_states_for_token_of_interest",
            50,
            "per_tag",
            "sliding_window",
        ),
        ("mcd", "save_hidden_states_given_token_idx", 1, "pooled", "langsam"),
        (
            "craft",
            "save_hidden_states_given_token_idx",
            1,
            "pooled",
            "sliding_window",
        ),
        (
            "coxlmm",
            "save_hidden_states_for_token_of_interest",
            100,
            "pooled",
            "none",
        ),
    ],
)
def test_method_definitions(
    pipeline, monkeypatch, tmp_path, method, hook, tokens, strategy, crop
):
    env = sourced_env("common", "coco", method)
    config = make_config(pipeline, monkeypatch, tmp_path, env)
    assert (
        config.hook_names,
        config.max_new_tokens,
        config.decomp_strategy,
        config.crop_mode,
    ) == (hook, tokens, strategy, crop)
    if crop == "langsam":
        assert config.concept_masks_per_image == 0  # class-free SAM masks
    if crop == "sliding_window":
        assert env["SLIDING_WINDOW_TAG_FILTER"] == "0"
    if method == "mcd":
        assert (
            env["MASK_FULL_IMAGE"],
            env["INPAINTING_METHOD"],
            env["MASK_CONTEXT_PIXELS"],
        ) == ("1", "blackout", "0")
    if method in ("mcd", "craft"):
        assert config.prompt_template == "null" and config.token_idx == -1
    # MCD uses its own decomposition (PCA); the others compare all four.
    assert [m.strip() for m in config.decomp_methods] == (
        ["pca"] if method == "mcd" else ["snmf", "simple", "sae2", "pca"]
    )
    # Only TGCL combines its concepts (symmetric alignment, noise removal);
    # the baselines stay as published.
    tgcl = method.startswith("tgcl")
    assert config.concept_sparsify == (("magnitude", 0.5) if tgcl else None)
    assert config.concept_ortho == ("lowdin" if tgcl else "")


def test_vocab_matches_choices():
    for dataset in DATASETS:
        env = sourced_env("common", dataset)
        vocab = (REPO / env["CONCEPTS_VOCAB"]).read_text().split("\n")
        assert sorted(v for v in vocab if v) == sorted(
            env["EXPL_CHOICES"].split(",")
        )
