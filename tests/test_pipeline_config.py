"""PipelineConfig defaults and step argument builders
(scripts/run_full_pipeline.py)."""

import importlib.util
from pathlib import Path
import pickle
import sys
import types

import pytest

REPO = Path(__file__).resolve().parent.parent

# Every variable PipelineConfig reads; cleared so the tests never depend on the
# caller's shell or on a .env file.
PIPELINE_ENV_VARS = [
    "BAG_SIZE",
    "BATCH_SIZE",
    "CLEAN_EXAMPLE_RATIO",
    "CONCEPT_CENTER",
    "CONCEPT_MASKS_PER_IMAGE",
    "CONCEPT_ORTHO",
    "CONCEPT_SPARSIFY",
    "CONCEPTS_VOCAB",
    "CROP_MODE",
    "DECOMP_COMPONENTS",
    "DECOMP_COMPONENTS_GLOBAL",
    "DECOMP_METHODS",
    "DECOMP_STRATEGY",
    "DELETE_INTERMEDIATE_FILES",
    "DETECTION_BATCH_SIZE",
    "DEVICE",
    "DEVICE_ID",
    "DL_ALPHA",
    "EVAL_TAG",
    "EXPL_CHOICES",
    "EXPL_BATCH_SIZE",
    "EXPL_INPUT_MODE",
    "EXPL_LABEL",
    "EXPL_MAX_IMAGES",
    "EXPL_PROMPT_MODE",
    "HF_HOME",
    "HOOK_NAMES",
    "IMAGE_BUDGET",
    "IMAGE_ROOT",
    "IMAGE_SIZE_HEIGHT",
    "IMAGE_SIZE_WIDTH",
    "INPUT_DIR",
    "LAYER_PATH",
    "LOGIT_LENS_LAYERS",
    "LOGIT_LENS_LAYER_SELECTION",
    "LOGIT_LENS_MODE",
    "LOGIT_LENS_NUM_PATCHES",
    "MASK_CONTEXT_PIXELS",
    "MASKS_PER_IMAGE",
    "MAX_IMAGES_PER_TAG",
    "MAX_NEW_TOKENS",
    "MIN_IMAGES_PER_TAG",
    "NUM_CONCEPT",
    "NUM_MOST_ACTIVATING_SAMPLES",
    "OUTPUT_DIR",
    "PATCH_SIZE",
    "POSITIVE_NEGATIVE_SEGMENT",
    "POSITIVE_NEGATVE_SEGMENT",
    "PROMPT",
    "PROMPT_TEMPLATE",
    "SAE_SPARSITY_LAMBDA",
    "SAE_TARGET_SPARSITY",
    "SEED",
    "SHUFFLE_CONCEPT_PROMPT",
    "SHUFFLE_CONCEPT_VOCAB",
    "SINGLE_OBJECT",
    "SLIDING_WINDOW_TAG_FILTER",
    "TOKEN_IDX",
    "TOP_N",
    "VLM_MODEL",
]


@pytest.fixture(scope="module")
def pipeline():
    """Import the pipeline script as a module (python-dotenv stubbed)."""
    sys.modules.setdefault(
        "dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
    )
    spec = importlib.util.spec_from_file_location(
        "run_full_pipeline", REPO / "scripts" / "run_full_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_config(pipeline, monkeypatch, tmp_path):
    def _make(**env):
        monkeypatch.setattr(pipeline, "ROOT_DIR", tmp_path)  # no .env there
        for key in PIPELINE_ENV_VARS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        return pipeline.PipelineConfig()

    return _make


# ---------------------------------------------------------------- C2: last
# token


def test_null_prompt_defaults_to_last_token_single_pass(pipeline, make_config):
    config = make_config(PROMPT_TEMPLATE="null")
    assert config.hook_names == pipeline.LAST_TOKEN_HOOK
    assert config.max_new_tokens == 1
    assert config.token_idx == -1
    args = pipeline.build_feature_extraction_args(config)
    assert args[args.index("--token_idx") + 1] == "-1"
    assert args[args.index("--max_new_tokens") + 1] == "1"


def test_last_token_hook_rejects_multi_step_generation(make_config):
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS=1"):
        make_config(PROMPT_TEMPLATE="null", MAX_NEW_TOKENS=5)


def test_concept_prompts_default_to_concept_span(make_config, pipeline):
    for template in ("cgdl", "llava", "non_contrastive"):
        config = make_config(PROMPT_TEMPLATE=template)
        assert config.hook_names == "save_hidden_states_for_token_of_interest"
        assert config.max_new_tokens == 50
        assert "--token_idx" not in pipeline.build_feature_extraction_args(
            config
        )


def test_class_free_masks_reject_fg_bg_mode(make_config):
    with pytest.raises(ValueError, match="CONCEPT_MASKS_PER_IMAGE=0"):
        make_config(
            CROP_MODE="langsam",
            CONCEPT_MASKS_PER_IMAGE=0,
            POSITIVE_NEGATIVE_SEGMENT=1,
        )
    config = make_config(CROP_MODE="langsam", CONCEPT_MASKS_PER_IMAGE=0)
    assert config.object_detector == "langsam"


def test_eval_tag_separates_evaluation_outputs(make_config, tmp_path):
    out = tmp_path / "out"
    untagged = make_config()
    assert (untagged.explain_dir, untagged.eval_dir) == (
        out / "explanations",
        out / "eval",
    )
    tagged = make_config(EVAL_TAG="grid")
    assert tagged.explain_dir == out / "explanations_grid"
    assert tagged.eval_dir == out / "eval_grid"
    assert tagged.plots_dir == out / "plots_grid"
    assert (
        tagged.decomp_dir == untagged.decomp_dir == out / "concept"
    )  # concepts are shared


def test_text_only_input_mode_validation(make_config):
    assert (
        make_config(
            EXPL_INPUT_MODE="text_only", SINGLE_OBJECT=1
        ).expl_input_mode
        == "text_only"
    )
    with pytest.raises(ValueError, match="SINGLE_OBJECT=1"):
        make_config(EXPL_INPUT_MODE="text_only")
    with pytest.raises(ValueError, match="EXPL_INPUT_MODE"):
        make_config(EXPL_INPUT_MODE="audio")


def test_explicit_mean_hook_with_null_prompt_still_allowed(make_config):
    """Stored rebuttal runs use P_null = null prompt + mean hook + 1 token."""
    config = make_config(
        PROMPT_TEMPLATE="null",
        HOOK_NAMES="save_hidden_states_mean",
        MAX_NEW_TOKENS=1,
    )
    assert config.hook_names == "save_hidden_states_mean"


# ------------------------------------------------- step 4: resume, no wipe


def test_interrupted_feature_step_resumes_instead_of_wiping(
    make_config, pipeline, monkeypatch
):
    """An interrupted step 4 keeps its finished tags (save_features skips them)
    and only clears the outputs that depend on the features."""
    import logging

    config = make_config(DECOMP_STRATEGY="per_tag")
    features = config.features_dir / "features"
    features.mkdir(parents=True)
    done = features / "save_hidden_states_for_token_of_interest_results_cat.pth"
    done.write_bytes(b"x")
    config.decomp_dir.mkdir(parents=True)
    (config.decomp_dir / "stale.pth").write_bytes(b"x")

    calls = []
    monkeypatch.setattr(
        pipeline, "_count_concepts_in_json", lambda path: 3
    )  # 1 of 3 tags done
    monkeypatch.setattr(
        pipeline,
        "_run_python_subprocess",
        lambda args, **kw: calls.append(args),
    )
    pipeline.step_4_generate_features(config, logging.getLogger("test"))

    assert done.exists()  # finished tag kept
    assert not config.decomp_dir.exists()  # downstream outputs cleared
    assert len(calls) == 1 and "tgcl.save_features" in calls[0]

    # All tags present: step skipped, nothing re-run.
    monkeypatch.setattr(pipeline, "_count_concepts_in_json", lambda path: 1)
    pipeline.step_4_generate_features(config, logging.getLogger("test"))
    assert len(calls) == 1


def test_explainer_batch_size(make_config):
    assert make_config().expl_batch_size == 1  # reproduces the published runs
    assert make_config(EXPL_BATCH_SIZE="16").expl_batch_size == 16
    with pytest.raises(ValueError, match="EXPL_BATCH_SIZE"):
        make_config(EXPL_BATCH_SIZE="0")


def test_concept_sparsify_setting(pipeline):
    assert pipeline.parse_concept_sparsify("") is None
    assert pipeline.parse_concept_sparsify("magnitude:0.25") == (
        "magnitude",
        0.25,
    )
    for bad in ("magnitude", "l1:0.5", "omp:0", "omp:2"):
        with pytest.raises(ValueError):
            pipeline.parse_concept_sparsify(bad)


def test_sparsify_step_keeps_the_dense_bank_and_runs_once(
    make_config, pipeline, monkeypatch
):
    import logging

    import torch

    config = make_config(
        CONCEPT_SPARSIFY="magnitude:0.25", DECOMP_METHODS="snmf"
    )
    bank = (
        config.decomp_dir
        / "snmf"
        / f"combined_concept_snmf_{config.ratio_tag}_raw.pth"
    )
    bank.parent.mkdir(parents=True)
    torch.save({"concepts": torch.ones(2, 4)}, bank)
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        torch.save(
            {
                "concepts": torch.eye(2, 4),
                "sparsification": {"algorithm": "magnitude", "density": 0.25},
            },
            args[args.index("--out") + 1],
        )

    monkeypatch.setattr(pipeline, "_run_python_subprocess", fake_run)
    log = logging.getLogger("test")
    assert pipeline._sparsify_concept_banks(config, log) is True
    dense = bank.with_name(bank.stem + "_dense.pth")
    assert torch.equal(
        torch.load(dense)["concepts"], torch.ones(2, 4)
    )  # dense original kept
    assert calls[0][calls[0].index("--bank") + 1] == str(
        dense
    )  # sparsified from the dense copy
    assert (
        pipeline._sparsify_concept_banks(config, log) is False
    )  # same setting: not again
    assert len(calls) == 1


def test_empty_bank_is_skipped_not_fatal(pipeline, make_config):
    import logging

    import torch

    config = make_config(DECOMP_METHODS="snmf,pca", CLEAN_EXAMPLE_RATIO="0.6")
    for method, n in (("snmf", 5), ("pca", 0)):
        d = config.decomp_dir / method
        d.mkdir(parents=True)
        torch.save(
            {"concepts": torch.zeros(n, 8) if n else torch.empty(0)},
            d / f"combined_concept_{method}_{config.ratio_tag}_raw.pth",
        )
    log = logging.getLogger("test")
    assert (
        pipeline._bank_size(
            config.decomp_dir
            / "pca"
            / f"combined_concept_pca_{config.ratio_tag}_raw.pth"
        )
        == 0
    )
    assert pipeline._active_methods(config, log) == ["snmf"]
    torch.save(
        {"concepts": torch.empty(0)},
        config.decomp_dir
        / "snmf"
        / f"combined_concept_snmf_{config.ratio_tag}_raw.pth",
    )
    with pytest.raises(RuntimeError, match="Every concept bank is empty"):
        pipeline._active_methods(config, log)


def test_bank_size_counts_missing_as_empty_and_rejects_corrupt_files(
    pipeline, tmp_path
):
    assert pipeline._bank_size(tmp_path / "missing.pth") == 0
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"not a torch file")
    with pytest.raises(pickle.UnpicklingError):
        pipeline._bank_size(corrupt)
