"""Step 5 entry point: concept banks from step-4 features, one tag at a time.

    python -m tgcl.analyse_features --features_path <tag features> ...

CPU decompositions of many tags run in parallel worker processes
(DECOMP_WORKERS); the grounding then runs per tag with the VLM's LM head.
"""

import copy
import gc
import multiprocessing
import os

from threadpoolctl import threadpool_limits
import torch

from tgcl.analysis import analyse_features
from tgcl.analysis import decompose_feature_file
from tgcl.device_utils import get_device_config
from tgcl.helpers.arguments import get_arguments
from tgcl.helpers.logger import log_args
from tgcl.helpers.logger import setup_logger
from tgcl.helpers.logit_lens import resolve_final_norm_module
from tgcl.helpers.utils import apply_determinism_setting
from tgcl.models import get_model_class


def cleanup_gpu_memory():
    """Clean up GPU memory by running garbage collection and clearing CUDA
    cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# CPU-only (sklearn) decompositions; these run in worker processes when there
# are many tags. sae* train with torch on the GPU and stay sequential.
PARALLEL_METHODS = {
    "snmf",
    "pca",
    "simple",
}


def concept_decomposition(args, model_subset, logger, decomposition=None):
    """tgcl.analysis.analyse_features for one tag's arguments (see there)."""
    analyse_features(
        analysis_name=args.analysis_name,
        logger=logger,
        model_class=model_subset,
        args=args,
        decomposition=decomposition,
    )


def decomp_workers() -> int:
    """Tags decomposed in parallel: DECOMP_WORKERS, else min(8, cpus // 4)."""
    value = os.environ.get("DECOMP_WORKERS", "").strip()
    return int(value) if value else max(1, min(8, (os.cpu_count() or 1) // 4))


def _single_thread():
    threadpool_limits(1)
    torch.set_num_threads(1)


def precompute_decompositions(all_args, workers: int, logger=None) -> dict:
    """{features file: decomposition} for every tag, computed by `workers`
    processes. Each tag is reseeded (decompose_tag), so the result is the one
    the sequential loop would compute. Must run before CUDA is initialised
    (fork)."""
    if (
        workers <= 1
        or len(all_args) <= 1
        or all_args[0].decomposition_method not in PARALLEL_METHODS
    ):
        return {}
    if logger is not None:
        logger.info(
            f"Decomposing {len(all_args)} tags with {workers} worker processes"
        )
    with multiprocessing.get_context("fork").Pool(
        workers, initializer=_single_thread
    ) as pool:
        results = pool.map(decompose_feature_file, all_args, chunksize=1)
    return {
        a.features_path[0]: r
        for a, r in zip(all_args, results)
        if r is not None
    }


def load_model_subset(args, logger, device, device_config) -> dict:
    """lm_head, tokenizer and final norm of the model; the model itself is
    freed."""
    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
        device_config=device_config,
    )

    # Extract only the components we need (lm_head and tokenizer)
    # Clone the lm_head weights to detach from the original model graph
    lm_head_weights = model_class.get_lm_head().weight.data.clone().float()
    lm_head_bias = None
    if model_class.get_lm_head().bias is not None:
        lm_head_bias = model_class.get_lm_head().bias.data.clone().float()

    # Create a standalone lm_head module
    lm_head = torch.nn.Linear(
        lm_head_weights.shape[1],
        lm_head_weights.shape[0],
        bias=(lm_head_bias is not None),
        device=device,
    )
    lm_head.weight.data = lm_head_weights.to(device)
    if lm_head_bias is not None:
        lm_head.bias.data = lm_head_bias.to(device)

    tokenizer = model_class.get_tokenizer()

    # Standalone copy of the model's final RMSNorm — needed to logit-lens
    # concepts extracted at mid decoder layers (per-tag layer selection):
    # lm_head on unnormalized residuals yields garbage top tokens.
    final_norm = None
    norm_module = resolve_final_norm_module(model_class.get_language_model())
    if norm_module is None:
        norm_module = resolve_final_norm_module(model_class.get_model())
    if norm_module is not None:
        final_norm = copy.deepcopy(norm_module).to(device).float().eval()

    # Delete the full model and clean GPU memory
    del model_class
    cleanup_gpu_memory()

    # Only the components the grounding needs
    return {
        "lm_head": lm_head,
        "tokenizer": tokenizer,
        "final_norm": final_norm,
    }


def main():
    """Main entry point for analyse_features script."""
    apply_determinism_setting()  # DETERMINISTIC=1
    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, "logs.log"))
    log_args(args, logger)

    device_config = get_device_config(args.device)
    device = device_config.primary_device
    logger.info(
        "Device config: %s -> primary=%s, gpu_ids=%s",
        device_config.raw,
        device,
        device_config.gpu_ids,
    )

    feature_source = args.features_path
    # Fail with a clear message instead of letting torch.load choke on a
    # missing path (a directory path that doesn't exist would otherwise fall
    # through to the file branch below).
    if (
        feature_source is not None
        and len(feature_source) == 1
        and not os.path.exists(feature_source[0])
    ):
        raise RuntimeError(
            f"--features_path '{feature_source[0]}' does not exist. "
            "The feature-generation step (save_features.py, pipeline step 4) "
            "did not produce any features — check its logs before re-running."
        )
    # if feature_source len is 1 and is not None and a dir path than a file path
    if (
        feature_source is not None
        and os.path.isdir(feature_source[0])
        and len(feature_source) == 1
    ):
        feature_source = feature_source[0]
        feature_files = os.listdir(feature_source)
        feature_files = [f for f in feature_files if f.endswith(".pth")]
        feature_files = [os.path.join(feature_source, f) for f in feature_files]
        if not feature_files:
            raise RuntimeError(
                f"--features_path directory '{feature_source}' contains "
                "no .pth "
                "feature files. The feature-generation step (save_features.py, "
                "pipeline step 4) did not save any features — check its logs."
            )
        concept_names = [
            args.save_filename + "_" + f.split("_")[-1].split(".")[0]
            for f in feature_files
        ]
        all_args = []

        for i, arg in enumerate(concept_names):
            arg = copy.deepcopy(args)
            arg.save_filename = concept_names[i]
            arg.features_path = [feature_files[i]]
            all_args.append(arg)

        runnable = []
        for arg in all_args:
            # Skip empty feature files (e.g. abstract concepts with 0
            # detections)
            check = torch.load(arg.features_path[0], map_location="cpu")
            if not check or len(check) == 0:
                logger.warning(
                    "Skipping empty feature file: %s "
                    "(concept '%s' had no detections)",
                    arg.features_path[0],
                    arg.save_filename,
                )
                continue
            # Per-tag layer selection: the feature file records which
            # layer produced its hidden states; decompose that module
            # instead of the static --module_to_decompose.
            selected_layer = check.get("selected_layer", None)
            if isinstance(selected_layer, (list, tuple)) and selected_layer:
                selected_layer = selected_layer[0]
            if isinstance(selected_layer, str) and selected_layer:
                if selected_layer != arg.module_to_decompose:
                    logger.info(
                        "Using selected layer from feature file for '%s': %s",
                        arg.save_filename,
                        selected_layer,
                    )
                arg.module_to_decompose = selected_layer
            # A tag can have a well-formed, non-empty file (real metadata
            # keys present) but still 0 assigned samples for this module
            # -- e.g. a small IMAGE_BUDGET where that tag had no hits.
            # decompose_activations()'s np.min/np.max stats print crashes
            # outright on an empty (0, hidden_dim) matrix, so this must be
            # caught here, before that call, not just the "file has no
            # keys at all" check above (which doesn't catch this case).
            hidden_states = check.get("hidden_states", {})
            n_samples = (
                len(hidden_states.get(arg.module_to_decompose, []))
                if isinstance(hidden_states, dict)
                else 0
            )
            del check
            if n_samples == 0:
                logger.warning(
                    "Skipping feature file with 0 samples for module '%s': %s "
                    "(concept '%s' had no assigned examples)",
                    arg.module_to_decompose,
                    arg.features_path[0],
                    arg.save_filename,
                )
                continue
            runnable.append(arg)

        decompositions = precompute_decompositions(
            runnable, decomp_workers(), logger
        )
        model_subset = load_model_subset(args, logger, device, device_config)
        for arg in runnable:
            concept_decomposition(
                arg,
                model_subset,
                logger,
                decompositions.get(arg.features_path[0]),
            )

    else:
        model_subset = load_model_subset(args, logger, device, device_config)
        concept_decomposition(args, model_subset, logger)

    # Final cleanup before exiting
    del model_subset
    cleanup_gpu_memory()
    logger.info("GPU memory cleaned up before exit.")


if __name__ == "__main__":
    main()
