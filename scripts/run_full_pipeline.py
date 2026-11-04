#!/usr/bin/env python3
"""
XL-VLMs Pipeline - Native Python Implementation

This script is a Python conversion of run_full_pipeline_without_coroping.sh
for easier debugging and development in VS Code.

Steps:
1) Dataset inference -> concepts map
2) Build crops JSON from concept→image mapping
3) (Optional, LOGIT_LENS_LAYER_SELECTION=1) Per-tag layer selection via
   logit lens -> logitlens/selected_layers.json
4) Generate features from crops JSON (on-the-fly cropping, hooks the
   per-tag selected layer when step 3 ran)
5) Decompose features (one or more methods)
6) Run VLM explainer per method
7) Concept deletion eval per method
8) (Optional) Plots per method + summary

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --input-dir /path/to/data \\
        --output-dir /path/to/output
"""

import argparse
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image
import torch

from tgcl.evaluation.concept_insertion_deletion import (
    SUMMARY as CONCEPT_CURVES_SUMMARY,
)
from tgcl.helpers.run_layout import eval_dir
from tgcl.helpers.run_layout import explanations_dir
from tgcl.helpers.run_layout import plots_dir
from tgcl.helpers.run_layout import validate_eval_tag
from tgcl.helpers.utils import resolve_layer_path
from tgcl.inference.dataset_inference import main as dataset_inference_main
from tgcl.preprocessing.concept_image_mapping import (
    main as concept_mapping_main,
)
from tgcl.preprocessing.crops_to_json import load_mapping
from tgcl.preprocessing.select_top_concepts import select_top_concepts
from tgcl.preprocessing.select_top_concepts import write_json_atomic
from tgcl.reporting.plot_concept_curves import plot_method
from tgcl.reporting.plot_concept_curves import plot_summary

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()

# Hook that reads the hidden state at one fixed input position (TOKEN_IDX).
LAST_TOKEN_HOOK = "save_hidden_states_given_token_idx"
# Explainer inputs: the evaluation image, or only the class name as text.
EXPL_INPUT_MODES = ("image", "text_only")

# =============================================================================
# Configuration
# =============================================================================


class PipelineConfig:
    """Configuration class that loads from .env file and CLI arguments."""

    def __init__(self):
        # Load .env file
        env_path = ROOT_DIR / ".env"
        if (
            env_path.exists() and os.environ.get("XLVLMS_DOTENV", "1") != "0"
        ):  # 0: run_experiments.py jobs
            load_dotenv(env_path)

        self.root_dir = ROOT_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Data & outputs
        # input_dir: Contains training images for object detection (step 2) and
        # feature generation (step 3)
        self.input_dir = self._get_path("INPUT_DIR", ROOT_DIR / "data")
        self.output_dir = self._get_path(
            "OUTPUT_DIR", ROOT_DIR / f"outputs/run_{self.timestamp}"
        )
        self.hf_home = self._get_path(
            "HF_HOME", Path.home() / ".cache/huggingface"
        )
        # image_root: Contains validation images for VLM explainer (step 5) -
        # separate from input_dir
        self.image_root = self._get_path("IMAGE_ROOT", self.input_dir / "grids")

        # Model/runtime
        self.vlm_model = self._get_str(
            "VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        self.batch_size = self._get_int("BATCH_SIZE", 48)
        # Step 1 runs full-size images; 24 GB GPUs need a smaller batch than the
        # crops of step 4.
        self.inference_batch_size = self._get_int(
            "INFERENCE_BATCH_SIZE", self.batch_size
        )
        self.seed = self._get_int("SEED", 42)
        # Device selection: DEVICE ("cuda:1", "cuda:0,2", "auto", "cpu");
        # DEVICE_ID=<n> is honored as a fallback alias when DEVICE is unset.
        self.device = self._get_str("DEVICE", "").strip()
        if not self.device:
            device_id = self._get_str("DEVICE_ID", "").strip()
            self.device = f"cuda:{device_id}" if device_id else "cuda:0"

        # Crops JSON generation
        self.masks_per_image = self._get_int("MASKS_PER_IMAGE", 5)
        self.concept_masks_per_image = self._get_int(
            "CONCEPT_MASKS_PER_IMAGE", 1
        )
        self.patch_size = self._get_int("PATCH_SIZE", 200)
        self.min_images_per_tag = self._get_int("MIN_IMAGES_PER_TAG", 10)
        self.max_images_per_tag = self._get_int("MAX_IMAGES_PER_TAG", 128)
        self.crop_mode = self._get_str("CROP_MODE", "random").strip().lower()
        if self.crop_mode in {"random", "sliding_window"}:
            self.object_detector = "none"
        elif self.crop_mode == "langsam":
            self.object_detector = "langsam"
        elif self.crop_mode == "none":
            # No real cropping: step 2 emits one whole-image bbox per (tag,
            # image) pair instead of running a detector, so steps 3-8 stay
            # completely unchanged (they only ever consume crops.json).
            self.object_detector = "none"
        else:
            raise ValueError(
                "Invalid CROP_MODE. Use one of: random, sliding_window, "
                "langsam, none"
            )
        self.detection_batch_size = self._get_int("DETECTION_BATCH_SIZE", 2)
        self.mask_context_pixels = self._get_int("MASK_CONTEXT_PIXELS", 0)
        # Positive/Negative binary segmentation (1=fg/bg only, 0=multi-mask)
        self.positive_negative_segment = int(
            os.environ.get(
                "POSITIVE_NEGATIVE_SEGMENT",
                os.environ.get("POSITIVE_NEGATVE_SEGMENT", "0"),
            )
        )
        # CONCEPT_MASKS_PER_IMAGE=0 with a detector = class-free masks (the
        # automatic mask generator only); fg/bg mode needs a tag mask.
        if (
            self.object_detector != "none"
            and self.concept_masks_per_image == 0
            and self.positive_negative_segment
        ):
            raise ValueError(
                "CONCEPT_MASKS_PER_IMAGE=0 (class-free masks) cannot be "
                "combined "
                "with POSITIVE_NEGATIVE_SEGMENT=1."
            )
        # Inference prompt and image preprocessing
        self.prompt = self._get_str(
            "PROMPT",
            "Identify every visible object, item, concept, and pattern in "
            "the image at the most fine-grained level. Output only single "
            "words in a strict comma-separated list, no sentences or "
            "explanations.",
        )
        self.prompt_template = self._get_str("PROMPT_TEMPLATE", "cgdl")
        # Decomposition strategy is an independent axis from PROMPT_TEMPLATE --
        # any template can use either:
        #   per_tag: one SNMF call per tag (DECOMP_COMPONENTS concepts/tag),
        #            merged + CLEAN_EXAMPLE_RATIO-purity-filtered by
        #            src/combine_concepts.py. Meaningful whenever each tag's
        #            features carry a real per-tag positive/negative signal
        #            to filter on.
        # pooled:  every tag's features concatenated once
        # (src/combine_features.py)
        #            into a single matrix, one SNMF call for the whole dataset
        #            (DECOMP_COMPONENTS_GLOBAL concepts total), no purity filter
        #            (every direction with >=1 assigned sample is accepted).
        # Defaults preserve each template's original behavior when
        # DECOMP_STRATEGY isn't set explicitly: cgdl (and cgdl_*) -> per_tag,
        # non_contrastive/null -> pooled. Set DECOMP_STRATEGY=per_tag/pooled
        # to override for any template (e.g. cgdl+pooled,
        # non_contrastive+per_tag).
        self.decomp_strategy = (
            self._get_str(
                "DECOMP_STRATEGY",
                "per_tag"
                if self.prompt_template.startswith("cgdl")
                else "pooled",
            )
            .strip()
            .lower()
        )
        if self.decomp_strategy not in ("per_tag", "pooled"):
            raise ValueError(
                "DECOMP_STRATEGY must be 'per_tag' or 'pooled', got "
                f"{self.decomp_strategy!r}"
            )
        # Use IMAGE_SIZE_WIDTH as the reference; height is derived per-image via
        # aspect ratio.
        self.image_size_width = self._get_int("IMAGE_SIZE_WIDTH", 512)
        # Keep for backward compatibility / debugging; not used as a fixed
        # resize target.
        self.image_size_height = self._get_int("IMAGE_SIZE_HEIGHT", 512)
        self.image_size = (self.image_size_width, self.image_size_height)
        self.image_budget = self._get_int(
            "IMAGE_BUDGET", 300
        )  # reduce for test , use 200 -> for a better run

        # Decomposition methods
        self.decomp_methods = self._get_str("DECOMP_METHODS", "snmf").split(",")
        self.decomp_components = self._get_int("DECOMP_COMPONENTS", 2)
        # Concept count for the non-contrastive/null global decomposition path
        # (step 5): these prompt templates don't produce a per-tag contrastive
        # fg/bg split, so all tags' features are pooled first and decomposed
        # once, budget-matched to roughly decomp_components * num_tags.
        self.decomp_components_global = self._get_int(
            "DECOMP_COMPONENTS_GLOBAL", 20
        )
        # Sparsity regularization: higher = sparser decomposition.
        # DL_ALPHA feeds SNMF/dictionary-learning method (snmf); it is
        # sklearn DictionaryLearning's L1 `alpha` on the sparse code (default
        # 20).
        # SAE_SPARSITY_LAMBDA feeds sae/sae2; it is the L1 penalty weight on the
        # SAE hidden code (default 0.0005). Try e.g. 1.0 for a much sparser
        # result.
        self.dl_alpha = self._get_float("DL_ALPHA", 20.0)
        # DL_DICT_ALPHA: L1 on the snmf concept vectors themselves (DL_ALPHA
        # only sparsifies the codes), as a fraction of the level at which a
        # concept vanishes (0 = off, the paper default).
        self.dl_dict_alpha = self._get_float("DL_DICT_ALPHA", 0.0)
        # CONCEPT_SPARSIFY=<algorithm>:<density> (e.g. magnitude:0.25): after
        # step 5 every
        # concept keeps only that fraction of its coordinates
        # (scripts/sparsify_concepts.py;
        # the dense bank is kept as *_raw_dense.pth). Empty = off.
        self.concept_sparsify = parse_concept_sparsify(
            self._get_str("CONCEPT_SPARSIFY", "")
        )
        # CONCEPT_CENTER=<alpha>: before that, subtract alpha times the bank
        # mean from every concept (1 = the whole component all concepts share,
        # which the explainer's matching removes anyway; it also carries part
        # of the prediction, so alpha trades overlap for faithfulness).
        # CONCEPT_ORTHO=lowdin: then orthogonalize the bank (Lowdin). Either
        # alone keeps every coordinate.
        self.concept_center = float(self._get_str("CONCEPT_CENTER", "0") or 0)
        self.concept_ortho = self._get_str("CONCEPT_ORTHO", "")
        if self.concept_ortho not in ("", "lowdin"):
            raise ValueError(f"CONCEPT_ORTHO={self.concept_ortho!r}")
        if (
            self.concept_center or self.concept_ortho
        ) and self.concept_sparsify is None:
            self.concept_sparsify = ("magnitude", 1.0)
        # concept-word alignment BERTScore inputs (eval/word_alignment_eval.py):
        # {image file: [captions]} and
        # LLaMA-3 phrases (eval/word_alignment_phrases.py); without both only
        # CLIPScore is computed.
        caps = os.environ.get("WORD_ALIGN_CAPTIONS_JSON", "")
        phr = os.environ.get("WORD_ALIGN_PHRASES_JSON", "")
        self.word_align_captions_json = (
            Path(self._expand_path(caps)) if caps else None
        )
        self.word_align_phrases_json = (
            Path(self._expand_path(phr)) if phr else None
        )
        self.sae_sparsity_lambda = self._get_float(
            "SAE_SPARSITY_LAMBDA", 0.0005
        )
        # Hard top-k activation for sae/sae2: guarantees this exact
        # zero-fraction
        # in the code (e.g. 0.99), unlike SAE_SPARSITY_LAMBDA which plateaus
        # well
        # short of high targets. Empty/unset disables it (soft L1 penalty only).
        sae_target_sparsity_value = self._get_str(
            "SAE_TARGET_SPARSITY", ""
        ).strip()
        self.sae_target_sparsity = (
            float(sae_target_sparsity_value)
            if sae_target_sparsity_value
            else None
        )
        self.num_concept = self._get_int(
            "NUM_CONCEPT", -1
        )  # -1 = use all tags, else top N tags by crop count
        # CONCEPT_MAP: a ready tag -> [image file] map (e.g. from dataset
        # labels); step 1 then
        # skips VLM tagging. CONCEPT_SHARD=k/K: this run handles every K-th tag
        # (k = 0..K-1)
        # of the selected map (steps 2-5 per shard;
        # scripts/merge_concept_shards.py joins them).
        map_value = os.environ.get("CONCEPT_MAP", "")
        self.concept_map_source = (
            Path(self._expand_path(map_value)) if map_value else None
        )
        self.concept_shard = parse_concept_shard(
            os.environ.get("CONCEPT_SHARD", "")
        )
        # Optional concept vocab filter. Unset/empty (e.g. commented out in
        # .env) => None => NUM_CONCEPT picks the top-N most frequent tags
        # without any vocab filtering.
        vocab_value = self._get_str("CONCEPTS_VOCAB", "").strip()
        self.concepts_vocab = Path(vocab_value) if vocab_value else None
        self.dataset_size = self._get_int("BAG_SIZE", 100)
        self.delete_intermediate_files = (
            self._get_int("DELETE_INTERMEDIATE_FILES", 0) == 1
        )

        # Explainer/Eval
        self.layer_path = resolve_layer_path(
            self._get_str("LAYER_PATH", "model.language_model.norm")
        )
        # Feature extracted per sample (HOOK_NAMES):
        #   save_hidden_states_for_token_of_interest (default): mean hidden
        #     state over the concept word's tokens in the generated answer
        #     (extract_token_of_interest_states); samples whose answer never
        #     names the concept are masked out.
        # save_hidden_states_given_token_idx (default for PROMPT_TEMPLATE=null,
        #     i.e. image-only input as in MCD/CRAFT): hidden state at input
        #     position TOKEN_IDX (-1 = last input token, the position that
        #     predicts the first answer token). Needs MAX_NEW_TOKENS=1, i.e.
        #     exactly one forward pass; with more steps the hook would see
        #     several concatenated passes.
        #   save_hidden_states_mean: mean over all generated tokens.
        default_hook_names = (
            LAST_TOKEN_HOOK
            if self.prompt_template == "null"
            else "save_hidden_states_for_token_of_interest"
        )
        self.hook_names = self._get_str("HOOK_NAMES", default_hook_names)
        self.token_idx = self._get_int("TOKEN_IDX", -1)
        # Per-tag layer selection via logit lens (step 3). When enabled, each
        # tag's extraction layer is picked by a logit-lens sweep and LAYER_PATH
        # is only the fallback; values flow to subprocesses via the environment.
        self.logit_lens_layer_selection = self._get_int(
            "LOGIT_LENS_LAYER_SELECTION", 0
        )
        self.logit_lens_mode = self._get_str("LOGIT_LENS_MODE", "patch")
        self.logit_lens_layers = self._get_str("LOGIT_LENS_LAYERS", "auto")
        self.logit_lens_num_patches = self._get_int("LOGIT_LENS_NUM_PATCHES", 8)

        self.top_n = self._get_int("TOP_N", 5)
        self.num_most_activating_samples = self._get_int(
            "NUM_MOST_ACTIVATING_SAMPLES", 10
        )
        # Tokens generated during feature extraction (steps 3/4). Default 50
        # (save_features.py's own CLI default); 1 for the last-token hook,
        # since --generation_mode with max_new_tokens=1 is exactly one forward
        # pass.
        uses_last_token_hook = self.hook_names.strip() == LAST_TOKEN_HOOK
        self.max_new_tokens = self._get_int(
            "MAX_NEW_TOKENS", 1 if uses_last_token_hook else 50
        )
        if uses_last_token_hook and self.max_new_tokens != 1:
            raise ValueError(
                f"HOOK_NAMES={LAST_TOKEN_HOOK} reads one forward pass and "
                "needs "
                f"MAX_NEW_TOKENS=1, got {self.max_new_tokens}."
            )
        self.expl_prompt_mode = self._get_str(
            "EXPL_PROMPT_MODE", "unsupervised"
        )
        self.expl_label = self._get_str("EXPL_LABEL", "")
        self.expl_choices = self._get_str("EXPL_CHOICES", "")
        # mcq choices from a file (one per line, e.g. a 1000-class vocabulary),
        # and the
        # number of random wrong choices asked next to each image's true
        # class(es)
        # (0 = every image gets the full list).
        choices_file = os.environ.get("EXPL_CHOICES_FILE", "")
        self.expl_choices_file = (
            Path(self._expand_path(choices_file)) if choices_file else None
        )
        self.expl_mcq_distractors = self._get_int("EXPL_MCQ_DISTRACTORS", 0)
        # Set when IMAGE_ROOT points at single-object crops rather than
        # multi-cell grids -- adjusts the explainer's mcq/unsupervised prompt
        # wording so it doesn't ask about non-existent grid cells.
        self.single_object = os.environ.get("SINGLE_OBJECT", "0") == "1"
        # Rebuttal P_bin_shuf control: substitute a different, reproducibly
        # sampled concept into the [concept] prompt placeholder instead of
        # the crop's true tag (see JSONDataset.create_dataset()). Only
        # affects step 4's prompt text -- concept-bank bucketing is untouched.
        self.shuffle_concept_prompt = (
            os.environ.get("SHUFFLE_CONCEPT_PROMPT", "0") == "1"
        )
        self.shuffle_concept_vocab = self._get_str("SHUFFLE_CONCEPT_VOCAB", "")
        # Images the explainer takes from IMAGE_ROOT:
        # -1 = all, 0 = skip explanation (and eval/plots), N = first N.
        self.expl_max_images = self._get_int("EXPL_MAX_IMAGES", -1)
        # Images per explainer generate() call. 1 reproduces the published runs;
        # 16 is ~2.7x faster but not bit-identical (left padding changes bf16
        # rounding: on 375 ImageNet images 99.7% identical answers, 98.8%
        # identical rank-1 concepts).
        self.expl_batch_size = self._get_int("EXPL_BATCH_SIZE", 1)
        if self.expl_batch_size < 1:
            raise ValueError(
                f"EXPL_BATCH_SIZE must be >= 1, got {self.expl_batch_size}"
            )
        # Positive/negative concept split threshold; embedded in the bank
        # file names (…_cr<ratio>_raw.pth) so runs with different ratios can
        # be evaluated side by side.
        self.clean_example_ratio = self._get_float("CLEAN_EXAMPLE_RATIO", 0.8)
        # Purity filtering is tied to DECOMP_STRATEGY, not PROMPT_TEMPLATE:
        # per_tag decomposition gives each tag's own positive/negative split a
        # real ratio to filter on (regardless of which template produced it),
        # while pooled decomposition has no such per-tag split to check --
        # every sample's hidden state is pooled together unfiltered (see
        # keep_only_token_of_interest in src/analysis/__init__.py), so
        # constraining concept selection to a CLEAN_EXAMPLE_RATIO threshold
        # meant for a per-tag split isn't meaningful there. Ratio 0 means no
        # filter: combine_concepts keeps every learned direction (all
        # DECOMP_COMPONENTS_GLOBAL of them), as the baselines are defined.
        # Written back to the environment so the combine_concepts.py
        # subprocess (which re-reads CLEAN_EXAMPLE_RATIO itself) and this
        # process's own ratio_tag-based file paths agree.
        if self.decomp_strategy == "pooled":
            self.clean_example_ratio = 0.0
            os.environ["CLEAN_EXAMPLE_RATIO"] = "0.0"
        self.ratio_tag = f"cr{self.clean_example_ratio:g}"

        # Evaluation variant (steps 6-8). EXPL_INPUT_MODE=text_only feeds the
        # explainer the class name as text, without the image (retrieval
        # "Text-only"); EVAL_TAG names the output directories of a variant.
        self.expl_input_mode = (
            self._get_str("EXPL_INPUT_MODE", "image").strip().lower()
        )
        if self.expl_input_mode not in EXPL_INPUT_MODES:
            raise ValueError(
                f"EXPL_INPUT_MODE must be one of {EXPL_INPUT_MODES}, got "
                f"{self.expl_input_mode!r}"
            )
        if self.expl_input_mode == "text_only" and not self.single_object:
            raise ValueError(
                "EXPL_INPUT_MODE=text_only needs SINGLE_OBJECT=1 (one "
                "class per image)."
            )
        self.eval_tag = validate_eval_tag(self._get_str("EVAL_TAG", ""))
        # EXPL_TOKEN_STATE=at: the explainer matches the state AT each answer
        # token to the concepts,
        # as TGCL's forced-answer features are taken (FORCED_RESPONSE); default:
        # the predicting state.
        self.expl_token_state = (
            self._get_str("EXPL_TOKEN_STATE", "predicting").strip().lower()
        )
        if self.expl_token_state not in ("predicting", "at"):
            raise ValueError(
                "EXPL_TOKEN_STATE must be 'predicting' or 'at', got "
                f"{self.expl_token_state!r}"
            )
        # EXPL_CENTER=calibration: centred concept matching, the answer-state
        # mean from
        # EXPL_CALIBRATION_PER_CLASS training images per class
        # (_calibration_list).
        self.expl_center = self._get_str("EXPL_CENTER", "").strip().lower()
        if self.expl_center not in ("", "calibration"):
            raise ValueError(
                "EXPL_CENTER must be empty or 'calibration', got "
                f"{self.expl_center!r}"
            )
        self.expl_calibration_per_class = self._get_int(
            "EXPL_CALIBRATION_PER_CLASS", 5
        )

        # Derived paths
        self._setup_derived_paths()

    def _expand_path(self, value: str) -> str:
        """Expand {ROOT_DIR} placeholder in paths."""
        return value.replace("{ROOT_DIR}", str(self.root_dir))

    def _get_str(self, key: str, default: str) -> str:
        value = os.environ.get(key, default)
        return self._expand_path(value) if "{ROOT_DIR}" in value else value

    def _get_int(self, key: str, default: int) -> int:
        value = os.environ.get(key)
        if value is None or value == "":
            return default
        return int(value)

    def _get_float(self, key: str, default: Optional[float]) -> Optional[float]:
        value = os.environ.get(key)
        if value is None or value == "":
            return default
        return float(value)

    def _get_path(self, key: str, default: Path) -> Path:
        value = os.environ.get(key)
        if value is None or value == "":
            return default
        expanded = self._expand_path(value)
        return Path(expanded)

    def _setup_derived_paths(self):
        """Setup derived paths based on output directory."""
        self.concept_map_json = (
            self.output_dir / "inference" / "concepts_to_images.json"
        )
        self.active_concept_map_json = self.concept_map_json
        self.crops_json = self.output_dir / "inference" / "crops.json"
        self.objects_csv = self.output_dir / "inference" / "objects.csv"
        self.logitlens_dir = self.output_dir / "logitlens"
        self.selected_layers_json = self.logitlens_dir / "selected_layers.json"
        self.features_dir = self.output_dir
        self.decomp_dir = self.output_dir / "concept"
        # Steps 6-8 outputs; EVAL_TAG keeps several evaluations of the same
        # concepts apart (see src/helpers/run_layout.py).
        self.explain_dir = explanations_dir(self.output_dir, self.eval_tag)
        self.eval_dir = eval_dir(self.output_dir, self.eval_tag)
        self.plots_dir = plots_dir(self.output_dir, self.eval_tag)
        self.logs_dir = self.output_dir / "logs"

    def update_from_args(self, args: argparse.Namespace):
        """Update config from CLI arguments."""
        if args.input_dir:
            self.input_dir = Path(args.input_dir)
        if args.output_dir:
            self.output_dir = Path(args.output_dir)
            self._setup_derived_paths()
        if args.decomp:
            self.decomp_methods = args.decomp.split(",")


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(logs_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("xl-vlms-pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate output when setup_logging is called multiple times.
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler
    log_file = (
        logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# GPU memory isolation helper
# =============================================================================


def _cleanup_gpu():
    """Best-effort in-process GPU memory cleanup.

    Only when this process already uses CUDA: the steps run in subprocesses, and
    torch.cuda.synchronize() in a process without a context creates one on the
    default device -- 256 MB on GPU 0 for every pipeline, whatever its DEVICE,
    which pushed other jobs on a shared GPU 0 out of memory.
    """
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_python_subprocess(
    script_args: List[str],
    env_overrides: dict = None,
    logger: logging.Logger = None,
) -> None:
    """Run a Python module/script in a fresh subprocess for full GPU memory
    isolation.

    This prevents CUDA OOM when consecutive pipeline steps each need the full
    GPU memory (e.g. Step 3 generate features → Step 4 decompose features).
    Captures stdout/stderr and relays through the pipeline logger so that
    combine_concepts skip summaries (and other subprocess output) appear in
    logs.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)

    # -u: unbuffered child stdout, so output streams live instead of arriving
    # in one block when the step finishes.
    cmd = [sys.executable, "-u"] + script_args
    if logger:
        logger.debug(f"Subprocess: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Relay subprocess output through the logger as it is produced
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if logger:
            logger.info(f"  [sub] {line}")
        else:
            print(line)

    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit {returncode}): {' '.join(cmd)}"
        )


# Metric inputs scripts/intermediates.py copies per method and evaluation
# variant into <run>/intermediate/.
VARIANT_INTERMEDIATES = (
    ("object_predictions", ".csv"),
    ("retrieval", ".csv"),
    ("concept_curves", ".json"),
    ("word_alignment", ".csv"),
)


def _variant_intermediates(run: Path, eval_tag: str) -> List[Path]:
    """The intermediate files of one evaluation variant, for every method (names
    as intermediates._suffixed)."""
    root = run / "intermediate"
    if not root.is_dir():
        return []
    return [
        d / (f"{name}_{eval_tag}{ext}" if eval_tag else f"{name}{ext}")
        for d in sorted(p for p in root.iterdir() if p.is_dir())
        for name, ext in VARIANT_INTERMEDIATES
    ]


def _downstream_outputs(
    config: PipelineConfig, step_num: int
) -> List[Tuple[Path, str]]:
    """What a re-computed step invalidates.

    Steps 1-5 change the concepts every evaluation variant reads: all variants'
    explanations, evaluations and plots go, with every intermediate. Steps 6 and
    7 only touch this variant (EVAL_TAG): its later outputs and its
    intermediates. Without this, re-running steps 5-7 left other variants'
    results, and paper_report's copies in <run>/intermediate/, stale.
    """
    run = config.output_dir
    out: List[Tuple[Path, str]] = []
    if step_num <= 1:
        out.append((config.crops_json, "Crops JSON"))
    if step_num <= 2:
        out.append((config.logitlens_dir, "Logit-lens layer selection"))
    if step_num <= 3:
        out.append((config.features_dir / "features", "Features"))
    if step_num <= 4:
        out.append((config.decomp_dir, "Decomposed concepts"))
    if step_num <= 5:
        for prefix, label in (
            ("explanations", "Explanations"),
            ("eval", "Evaluations"),
            ("plots", "Plots"),
        ):
            out += [
                (p, f"{label} ({p.name})")
                for p in sorted(run.glob(f"{prefix}*"))
                if p.is_dir()
            ]
        out.append((run / "intermediate", "Intermediates"))
    elif step_num <= 7:
        if step_num == 6:
            out.append((config.eval_dir, "Evaluations"))
        out.append((config.plots_dir, "Plots"))
        out += [
            (p, f"Intermediate ({p.parent.name}/{p.name})")
            for p in _variant_intermediates(run, config.eval_tag)
        ]
    return out


def _delete_downstream_outputs(
    config: PipelineConfig, step_num: int, logger: logging.Logger
) -> None:
    """Delete what a re-computed step invalidates (_downstream_outputs), so it
    is re-computed too."""
    downstream_deletions = _downstream_outputs(config, step_num)
    if downstream_deletions:
        logger.info(
            f"Step {step_num} was re-computed; cascade deleting "
            "downstream outputs..."
        )
        for path, label in downstream_deletions:
            if path.exists():
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink()
                    logger.info(f"  Deleted: {label} ({path})")
                except OSError as e:
                    logger.warning(f"  Could not delete {label}: {e}")


def parse_concept_sparsify(value: str):
    """'magnitude:0.25' -> ('magnitude', 0.25); '' -> None."""
    value = (value or "").strip()
    if not value:
        return None
    algorithm, _, density = value.partition(":")
    if algorithm not in ("magnitude", "omp") or not density:
        raise ValueError(
            "CONCEPT_SPARSIFY must be magnitude:<density> or "
            f"omp:<density>, got {value!r}"
        )
    d = float(density)
    if not 0.0 < d <= 1.0:
        raise ValueError(f"CONCEPT_SPARSIFY density must be in (0, 1], got {d}")
    return algorithm, d


def parse_concept_shard(value: str) -> Optional[Tuple[int, int]]:
    """'k/K' -> (k, K) with 0 <= k < K; '' -> None."""
    if not value:
        return None
    k, K = (int(x) for x in value.split("/"))
    if not 0 <= k < K:
        raise ValueError(
            f"CONCEPT_SHARD must be k/K with 0 <= k < K, got {value!r}"
        )
    return k, K


def shard_concept_map(concept_map_json: Path, shard: Tuple[int, int]) -> Path:
    """Write tags with sorted index % K == k to <stem>_shard{k}of{K}.json."""
    k, K = shard
    with open(concept_map_json, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    keep = {
        tag: mapping[tag] for i, tag in enumerate(sorted(mapping)) if i % K == k
    }
    out = concept_map_json.with_name(
        f"{concept_map_json.stem}_shard{k}of{K}.json"
    )
    write_json_atomic(out, keep, indent=1)
    return out


def _active_concept_map(config: PipelineConfig, logger: logging.Logger) -> Path:
    """The tag -> images map steps 2-5 use: vocab-filtered top NUM_CONCEPT tags,
    restricted to this run's CONCEPT_SHARD."""
    result = select_top_concepts(
        config.concept_map_json, config.num_concept, config.concepts_vocab
    )
    if result != config.concept_map_json:
        logger.info(f"Using filtered concept map: {result}")
    if config.concept_shard is not None:
        result = shard_concept_map(result, config.concept_shard)
        logger.info(
            "Concept shard "
            f"{config.concept_shard[0]}/{config.concept_shard[1]}: "
            f"{_count_concepts_in_json(result)} tags ({result})"
        )
    return result


def _count_concepts_in_json(concept_map_json: Path) -> int:
    if not concept_map_json.exists():
        return 0
    with open(concept_map_json, "r", encoding="utf-8") as handle:
        concept_mapping = json.load(handle)
    if isinstance(concept_mapping, dict):
        return len(concept_mapping)
    return 0


# =============================================================================
# Pipeline Steps
# =============================================================================


def create_directories(config: PipelineConfig, logger: logging.Logger):
    """Create all necessary output directories."""
    dirs = [
        config.logs_dir,
        config.output_dir / "inference",
        config.output_dir / "features",
        config.output_dir / "concept",
        config.output_dir / "explanations",
        config.output_dir / "eval",
        config.output_dir / "plots",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directories under {config.output_dir}")


def step_1_dataset_inference(config: PipelineConfig, logger: logging.Logger):
    """Step 1: Dataset inference -> concept map."""

    config.objects_csv.parent.mkdir(parents=True, exist_ok=True)

    if config.concept_map_source is not None:
        # Tags come from the dataset (e.g. ImageNet labels), not from VLM
        # descriptions.
        if not config.concept_map_json.exists():
            shutil.copyfile(config.concept_map_source, config.concept_map_json)
            logger.info(
                f"Concept map from {config.concept_map_source} (no VLM tagging)"
            )
        config.active_concept_map_json = _active_concept_map(config, logger)
        return

    # Check if both CSV and concept map exist before skipping
    csv_exists = config.objects_csv.exists()
    concept_map_exists = config.concept_map_json.exists()

    if csv_exists and concept_map_exists:
        logger.info(
            f"Skip Step 1 (found {config.objects_csv} and "
            f"{config.concept_map_json})"
        )
        # Still resolve the filtered map so later steps log/count correctly
        config.active_concept_map_json = _active_concept_map(config, logger)
        return

    # Step will re-compute; cascade delete downstream outputs
    _delete_downstream_outputs(config, 1, logger)

    if not csv_exists:
        # Import and run dataset_inference
        logger.info("START: Dataset Inference")

        # Prepare arguments as if from command line
        inference_args = [
            "--dataset_path",
            str(config.input_dir),
            "--model_name",
            config.vlm_model,
            "--output_csv",
            str(config.objects_csv),
            "--prompt",
            config.prompt,
            "--batch_size",
            str(config.inference_batch_size),
            "--image_size_width",
            str(config.image_size_width),
            "--image_budget",
            str(config.image_budget),
            "--device",
            config.device,
            "--trust_remote_code",
        ]

        dataset_inference_main(inference_args)

        logger.info("DONE: Dataset Inference")
    else:
        logger.info(f"Reusing existing CSV: {config.objects_csv}")

    # Build concept map
    if not concept_map_exists:
        logger.info("START: Build Concept Map")

        mapping_args = [
            "--input",
            str(config.objects_csv),
            "--output",
            str(config.concept_map_json),
        ]

        concept_mapping_main(mapping_args)

        logger.info("DONE: Build Concept Map")
    else:
        logger.info(f"Reusing existing concept map: {config.concept_map_json}")

    config.active_concept_map_json = _active_concept_map(config, logger)


def _build_whole_image_crops_json(
    config: PipelineConfig, logger: logging.Logger, concept_map_json: Path
) -> None:
    """CROP_MODE=none variant of step 2: instead of running a detector,
    emit one crops.json entry per (tag, image) with bbox = the full image
    extent. Steps 3-8 need no changes since they only ever consume
    crops.json in this same schema (bbox-only records, RLE off, are already
    a supported variant — see RLE=0 in .env.example).
    """

    if config.seed is not None:
        random.seed(config.seed)

    mapping = load_mapping(str(concept_map_json))

    valid_tags = []
    for tag, rels in mapping.items():
        if not isinstance(rels, list) or len(rels) < config.min_images_per_tag:
            continue
        if (
            config.max_images_per_tag > 0
            and len(rels) > config.max_images_per_tag
        ):
            rels = random.sample(rels, config.max_images_per_tag)
        valid_tags.append((tag, rels))

    if not valid_tags:
        raise RuntimeError(
            f"No concept tag survived filtering: all {len(mapping)} tags in "
            f"'{concept_map_json}' have fewer than min_images_per_tag="
            f"{config.min_images_per_tag} images."
        )

    total_images = sum(len(rels) for _, rels in valid_tags)
    logger.info(
        f"Building whole-image crops.json: {total_images} images across "
        f"{len(valid_tags)} tags"
    )

    result = {}
    skipped = 0
    for tag, rels in valid_tags:
        tag_entry = {}
        for rel in rels:
            image_path = config.input_dir / rel
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except OSError as e:  # missing, or not an image PIL can read
                logger.warning(f"  Skipping unreadable image {image_path}: {e}")
                skipped += 1
                continue
            tag_entry[rel] = {
                "masks_rle": [
                    {"bbox": [0, 0, width, height], "is_concept": True}
                ]
            }
        if tag_entry:
            result[tag] = tag_entry

    if not result:
        raise RuntimeError(
            "Whole-image crops.json build produced zero entries (all "
            "images unreadable?)."
        )

    config.crops_json.parent.mkdir(parents=True, exist_ok=True)
    with open(config.crops_json, "w", encoding="utf-8") as f:
        json.dump(result, f)
    logger.info(
        f"Wrote whole-image crops.json: {config.crops_json} "
        f"({total_images - skipped} images, {skipped} skipped)"
    )


def step_2_build_crops_json(config: PipelineConfig, logger: logging.Logger):
    """Step 2: Build crops JSON from concept→image map.

    Runs as a subprocess to isolate detector GPU memory (skipped for
    CROP_MODE=none, which builds crops.json in-process — no detector to load).
    """

    if config.crops_json.exists():
        # A crops.json without any concept entries (e.g. from a run where
        # every tag was filtered out) must not be reused — it would make
        # every downstream step fail with "0 concept splits".
        try:
            with open(config.crops_json, "r", encoding="utf-8") as handle:
                crops_value = json.load(handle)
        except (OSError, ValueError):  # unreadable or not JSON: rebuild it
            crops_value = None
        if isinstance(crops_value, dict) and len(crops_value) > 0:
            logger.info(f"Skip Step 2 (found {config.crops_json})")
            return
        logger.warning(
            f"Existing {config.crops_json} is empty or unreadable — "
            "rebuilding it."
        )

    # Step will re-compute; cascade delete downstream outputs
    _delete_downstream_outputs(config, 2, logger)

    logger.info("START: Crops JSON")
    config.crops_json.parent.mkdir(parents=True, exist_ok=True)
    concept_map_json = _active_concept_map(config, logger)

    if config.crop_mode == "none":
        _build_whole_image_crops_json(config, logger, concept_map_json)
        logger.info("DONE: Crops JSON (whole-image, no cropping)")
        return

    crops_args = [
        "-m",
        "tgcl.preprocessing.crops_to_json",
        "--mapping_json",
        str(concept_map_json),
        "--image_root",
        str(config.input_dir),
        "--output_json",
        str(config.crops_json),
        "--detector",
        config.object_detector,
        "--masks_per_image",
        str(config.masks_per_image),
        "--concept_masks_per_image",
        str(config.concept_masks_per_image),
        "--min_images_per_tag",
        str(config.min_images_per_tag),
        "--max_images_per_tag",
        str(config.max_images_per_tag),
        "--patch_size",
        str(config.patch_size),
        "--batch_size",
        str(config.detection_batch_size),
        "--image_size_width",
        str(config.image_size_width),
        "--device",
        config.device,
        "--seed",
        str(config.seed),
        "--positive_negative_segment",
        str(config.positive_negative_segment),
    ]

    _run_python_subprocess(crops_args, logger=logger)

    logger.info("DONE: Crops JSON")


def step_3_select_layers(config: PipelineConfig, logger: logging.Logger):
    """Step 3: Per-tag layer selection via logit lens (optional).

    Runs only when LOGIT_LENS_LAYER_SELECTION=1. For each concept tag in
    crops.json, sweeps decoder layers on sampled tag regions and writes the
    winning layer per tag to logitlens/selected_layers.json, which step 4 uses
    to hook the right layer during feature extraction.

    Skipped whenever DECOMP_STRATEGY=pooled, for ANY prompt template: pooled
    decomposition concatenates every tag's features and decomposes once (step
    5), which requires every tag's hidden states to live under the SAME module
    key. A per-tag layer would break that (and does: mixed module keys after
    pooling fail step 5's fixed --module_to_decompose). per_tag decomposition
    has no such constraint -- each tag is decomposed separately, so a different
    layer per tag is fine -- which is why this is keyed off decomp_strategy, not
    prompt_template: it's universal across templates for a given strategy, not
    special-cased per template.

    Runs as a subprocess to isolate VLM GPU memory.
    """
    if not config.logit_lens_layer_selection:
        logger.info("Skip Step 3 (LOGIT_LENS_LAYER_SELECTION=0)")
        return
    if config.decomp_strategy == "pooled":
        logger.info(
            "Skip Step 3 (decomp_strategy=pooled uses a uniform layer, "
            "not per-tag selection; "
            f"prompt_template={config.prompt_template})"
        )
        return

    if config.selected_layers_json.exists():
        logger.info(f"Skip Select Layers (found {config.selected_layers_json})")
        return

    # Step will re-compute; cascade delete downstream outputs
    _delete_downstream_outputs(config, 3, logger)

    logger.info("START: Select Layers (logit lens)")

    select_args = [
        "-m",
        "tgcl.select_layers",
        "--model_name",
        config.vlm_model,
        "--dataset_name",
        "json_crop_map",
        "--dataset_size",
        str(config.dataset_size),
        "--data_dir",
        str(config.input_dir),
        "--annotation_file",
        str(config.crops_json),
        "--split",
        "train",
        "--prompt_template",
        config.prompt_template,
        "--save_dir",
        str(config.features_dir),
        "--batch_size",
        "1",
        "--generation_mode",
        "--max_new_tokens",
        str(config.max_new_tokens),
    ]

    _run_python_subprocess(select_args, logger=logger)

    logger.info("DONE: Select Layers (logit lens)")


def step_4_generate_features(config: PipelineConfig, logger: logging.Logger):
    """Step 4: Generate features from crops JSON.

    Runs as a subprocess to isolate VLM GPU memory from Step 5. Works unchanged
    for CROP_MODE=none too, since crops.json already has one whole-image bbox
    per (tag, image) from step 2 in that mode.
    """

    features_path = config.features_dir / "features"
    selected_concept_count = _count_concepts_in_json(
        config.active_concept_map_json
    )
    existing_pth_count = (
        len(list(features_path.glob("*.pth"))) if features_path.exists() else 0
    )
    # Require the full expected tag count, not just "at least one file" --
    # a config interrupted mid-loop (e.g. a killed subprocess) can leave
    # exactly one tag's .pth file on disk, which the old any(...) check
    # treated as fully done, silently decomposing/evaluating on 1 of N tags
    # for the rest of the pipeline with no error raised. Confirmed this
    # happened in practice: a killed run left only one tag's feature file,
    # and every step downstream skip-checked past it to a "successful"
    # completion built on 10% of the intended data.
    features_complete = (
        features_path.exists()
        and existing_pth_count > 0
        and (
            selected_concept_count <= 0
            or existing_pth_count >= selected_concept_count
        )
    )
    if features_complete:
        logger.info(
            f"Skip Generate Features (found {existing_pth_count} "
            f"feature files under {features_path})"
        )
        return
    if existing_pth_count > 0:
        # save_features.py writes each tag's file atomically and skips tags
        # whose file exists, so an interrupted run resumes where it stopped.
        # (Re-running steps 1-3 deletes the features, so these files always
        # come from the current crops.)
        logger.warning(
            f"Found {existing_pth_count} feature file(s) under "
            f"{features_path} but expected "
            f"{selected_concept_count} (from the filtered concept "
            "map) -- resuming the "
            "interrupted run with the missing tags."
        )
    # Step will re-compute; cascade delete downstream outputs
    _delete_downstream_outputs(config, 4, logger)

    logger.info("START: Generate Features")
    if selected_concept_count > 0:
        logger.info(
            f"Using top {selected_concept_count} concepts from the "
            "filtered concept map for feature generation"
        )

    features_args = build_feature_extraction_args(config)

    # Force off regardless of the .env value: pooled decomposition (any
    # template) concatenates every tag's features and decomposes once against
    # one fixed module path, so save_features.py's per-tag layer-selection
    # fallback (which would hook a different layer per tag) must not run.
    step4_env_overrides = None
    if config.decomp_strategy == "pooled":
        step4_env_overrides = {"LOGIT_LENS_LAYER_SELECTION": "0"}

    _run_python_subprocess(
        features_args, env_overrides=step4_env_overrides, logger=logger
    )

    logger.info("DONE: Generate Features")


def build_feature_extraction_args(config: PipelineConfig) -> List[str]:
    """Command line for src/save_features.py (step 4)."""
    args = [
        "-m",
        "tgcl.save_features",
        "--model_name",
        config.vlm_model,
        "--dataset_name",
        "json_crop_map",
        "--dataset_size",
        str(config.dataset_size),
        "--data_dir",
        str(config.input_dir),
        "--annotation_file",
        str(config.crops_json),
        "--split",
        "train",
        "--hook_names",
        config.hook_names,
        "--modules_to_hook",
        config.layer_path,
        "--prompt_template",
        config.prompt_template,
        "--save_dir",
        str(config.features_dir),
        "--batch_size",
        str(config.batch_size),
        "--generation_mode",
        "--max_new_tokens",
        str(config.max_new_tokens),
        "--save_only_generated_tokens",
        "--exact_match_modules_to_hook",
    ]
    if config.hook_names.strip() == LAST_TOKEN_HOOK:
        args.extend(["--token_idx", str(config.token_idx)])
    if config.shuffle_concept_prompt and config.shuffle_concept_vocab:
        args.append("--shuffle_concept_prompt")
        args.extend(["--shuffle_concept_vocab", config.shuffle_concept_vocab])
    return args


def step_5_decompose_features(config: PipelineConfig, logger: logging.Logger):
    """Step 5: Decompose features across methods.

    Each analyse_features call runs as a subprocess to isolate GPU memory, since
    loading the model's lm_head requires significant VRAM. Works unchanged for
    CROP_MODE=none too — features_dir/"features" still holds one .pth per tag
    (built from whole-image crops in step 4).

    DECOMP_STRATEGY selects which of the two happens, independent of
    PROMPT_TEMPLATE: per_tag decomposes each tag's features separately (below)
    and merges; pooled concatenates every tag's features into one matrix first
    and decomposes once, with each resulting atom used directly as a concept --
    see combine_features.py.
    """
    use_per_tag_decomp = config.decomp_strategy == "per_tag"

    step_4_recomputed = False  # Track if any method was re-computed

    for method in config.decomp_methods:
        method = method.strip()
        method_dir = config.decomp_dir / method
        out_raw = (
            method_dir / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        )

        if out_raw.exists():
            logger.info(f"Skip Decompose ({method}) (found {out_raw})")
            continue

        # Mark that step 4 is being re-computed for this method
        step_4_recomputed = True

        method_dir.mkdir(parents=True, exist_ok=True)

        # Analyse features (subprocess — loads model for lm_head)
        base_analysis_name = (
            "decompose_activations_text_grounding_image_grounding"
        )
        intermediate_dir = method_dir / f"intermediate_{method}"

        logger.info(f"START: Decompose:{method} (batch)")

        features_dir = config.features_dir / "features"
        if use_per_tag_decomp:
            decompose_features_path = str(features_dir)
            decompose_num_concepts = config.decomp_components
        else:
            # Pool every tag's features into a single combined_features.pth
            # (once) so analyse_features.py takes its single-matrix path
            # instead of looping per tag.
            combined_features_path = features_dir / "combined_features.pth"
            if not combined_features_path.exists():
                _run_python_subprocess(
                    [
                        "-m",
                        "tgcl.combine_features",
                        str(features_dir),
                    ],
                    logger=logger,
                )
            decompose_features_path = str(combined_features_path)
            decompose_num_concepts = config.decomp_components_global

        analyse_args = [
            "-m",
            "tgcl.analyse_features",
            "--model_name",
            config.vlm_model,
            "--analysis_name",
            f"{base_analysis_name}_{method}",
            "--features_path",
            decompose_features_path,
            "--module_to_decompose",
            config.layer_path,
            "--num_concepts",
            str(decompose_num_concepts),
            "--decomposition_method",
            method,
            "--num_most_activating_samples",
            str(config.num_most_activating_samples),
            "--save_dir",
            str(intermediate_dir),
            "--dl_alpha",
            str(config.dl_alpha),
            "--dl_dict_alpha",
            str(config.dl_dict_alpha),
            "--sae_sparsity_lambda",
            str(config.sae_sparsity_lambda),
        ]
        if config.sae_target_sparsity is not None:
            analyse_args += [
                "--sae_target_sparsity",
                str(config.sae_target_sparsity),
            ]

        _run_python_subprocess(analyse_args, logger=logger)

        logger.info(f"DONE: Decompose:{method} (batch)")

        # Combine concepts (lightweight, no GPU — runs in-process)
        logger.info(f"START: Combine Concepts ({method})")

        combine_cli_args = [
            "-m",
            "tgcl.combine_concepts",
            "--input_dir",
            str(intermediate_dir),
            "--output_path",
            str(
                method_dir / f"combined_concept_{method}.pth"
            ),  # combine appends _cr<ratio>
            "--normalization",
            "gl",
        ]
        if config.delete_intermediate_files:
            combine_cli_args.append("--delete")

        _run_python_subprocess(combine_cli_args, logger=logger)

        logger.info(f"DONE: Combine Concepts ({method})")

        # An empty bank (no concept passed the purity filter) is a result for
        # this
        # method: it is marked and skipped in steps 6-8; the run fails only when
        # every method is empty (_active_methods).
        raw_bank_path = (
            method_dir / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        )
        n_concepts = _bank_size(raw_bank_path)
        if n_concepts == 0:
            message = (
                f"Combined concept bank is EMPTY ({raw_bank_path}). No concept "
                "passed the "
                "CLEAN_EXAMPLE_RATIO="
                f"{os.environ.get('CLEAN_EXAMPLE_RATIO', '0.8')} "
                "purity filter — the VLM answered 'No [tag]' for the "
                "top-activating "
                "regions of every direction. All filtered concepts were "
                "saved to the "
                "negative bank "
                f"(combined_negative_concept_{method}_{config.ratio_tag}"
                "_*.pth) "
                "for inspection. "
                "Check the 'Combine Concepts' log above for per-concept "
                "positive ratios; "
                "consider lowering CLEAN_EXAMPLE_RATIO, increasing "
                "regions per tag, or "
                "reviewing the crops (CROP_MODE, PATCH_SIZE)."
            )
            (method_dir / EMPTY_BANK_MARKER).write_text(message + "\n")
            logger.warning(
                f"{message} Method {method} is skipped in steps 6-8."
            )
            if intermediate_dir.exists():
                shutil.rmtree(intermediate_dir, ignore_errors=True)
            continue
        logger.info(
            f"Combined concept bank: {n_concepts} concepts ({raw_bank_path})"
        )

        # Regrounding (subprocess — loads model again)
        logger.info(f"START: Reground Concepts ({method})")

        reground_args = [
            "-m",
            "tgcl.analyse_features",
            "--model_name",
            config.vlm_model,
            "--analysis_name",
            f"redefine_activations_text_grounding_{method}",
            "--analysis_saving_path",
            str(
                method_dir
                / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
            ),
            "--module_to_decompose",
            config.layer_path,
            "--decomposition_method",
            method,
            "--save_filename",
            f"combined_concept_{method}_{config.ratio_tag}_gl_regrounded",
            "--save_dir",
            str(method_dir),
            "--load_matched_features",
        ]

        _run_python_subprocess(reground_args, logger=logger)

        logger.info(f"DONE: Reground Concepts ({method})")

        # Cleanup intermediate directory
        if intermediate_dir.exists():
            shutil.rmtree(intermediate_dir, ignore_errors=True)

    _active_methods(config, logger)  # fails if every bank is empty
    # If any method was re-computed, cascade delete downstream outputs
    if _sparsify_concept_banks(config, logger):
        step_4_recomputed = True
    if step_4_recomputed:
        _delete_downstream_outputs(config, 5, logger)


def _sparsify_concept_banks(
    config: PipelineConfig, logger: logging.Logger
) -> bool:
    """Apply CONCEPT_SPARSIFY to every method's combined bank; True if a bank
    changed."""
    if config.concept_sparsify is None:
        return False
    algorithm, density = config.concept_sparsify
    changed = False
    for method in (m.strip() for m in config.decomp_methods):
        bank = (
            config.decomp_dir
            / method
            / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        )
        if not bank.exists() or _bank_size(bank) == 0:
            continue
        done = (
            torch.load(bank, map_location="cpu", weights_only=False).get(
                "sparsification"
            )
            or {}
        )
        if (
            done.get("algorithm"),
            done.get("density"),
            float(done.get("center") or 0),
            done.get("orthogonalize") or "",
        ) == (algorithm, density, config.concept_center, config.concept_ortho):
            logger.info(
                f"Skip Sparsify ({method}): already {algorithm} {density}"
            )
            continue
        dense = bank.with_name(bank.stem + "_dense.pth")
        if not dense.exists():
            shutil.copyfile(bank, dense)
        logger.info(
            f"START: Sparsify concepts ({method}, {algorithm}, density "
            f"{density})"
        )
        _run_python_subprocess(
            [
                str(ROOT_DIR / "scripts" / "sparsify_concepts.py"),
                "--run",
                str(config.features_dir),
                "--method",
                method,
                "--bank",
                str(dense),
                "--density",
                str(density),
                "--algorithm",
                algorithm,
                "--center",
                str(config.concept_center),
                "--orthogonalize",
                config.concept_ortho,
                "--out",
                str(bank),
                "--reground",
                "--model_name",
                config.vlm_model,
                "--layer",
                config.layer_path,
            ],
            logger=logger,
        )
        logger.info(f"DONE: Sparsify concepts ({method})")
        changed = True
    return changed


EMPTY_BANK_MARKER = "EMPTY_BANK.txt"


def _bank_size(path: Path) -> int:
    """Number of concepts in a combined bank (0 if missing or empty; a
    corrupt file raises)."""
    if not Path(path).exists():
        return 0
    concepts = torch.load(path, map_location="cpu", weights_only=False).get(
        "concepts"
    )
    return int(concepts.shape[0]) if getattr(concepts, "ndim", 0) == 2 else 0


def _active_methods(
    config: PipelineConfig, logger: logging.Logger
) -> List[str]:
    """Decomposition methods with a non-empty concept bank (steps 6-8); fails if
    there is none."""
    active, empty = [], []
    for method in (m.strip() for m in config.decomp_methods):
        bank = (
            config.decomp_dir
            / method
            / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        )
        (active if _bank_size(bank) > 0 else empty).append(method)
    if empty:
        logger.warning(
            f"No concepts (empty or missing bank): {', '.join(empty)} — "
            "skipped in steps 6-8"
        )
    if not active:
        raise RuntimeError(
            f"Every concept bank is empty or missing ({', '.join(empty)}); see "
            f"{config.decomp_dir}/<method>/{EMPTY_BANK_MARKER}"
        )
    return active


def _calibration_list(config: PipelineConfig, logger: logging.Logger) -> Path:
    """EXPL_CENTER=calibration: the training images the explainer takes its
    answer-state mean from
    (inference.vlm_explainer_multibatch.VLMConceptExplainer.set_centering), as
    JSON in the variant's
    explanations folder: EXPL_CALIBRATION_PER_CLASS images per class of the
    concept map steps 2-5 used,
    with their labels -- or, for grid inputs (SINGLE_OBJECT=0), 2x2 grids of
    them (composed in memory).
    Deterministic (SEED); no test image is used."""
    concept_map = json.load(
        open(_active_concept_map(config, logger), encoding="utf-8")
    )
    rng = random.Random(config.seed)
    per_class = {}
    for cls in sorted(concept_map):
        files = sorted(concept_map[cls])
        picked = rng.sample(
            files, min(config.expl_calibration_per_class, len(files))
        )
        per_class[cls] = [str(config.input_dir / f) for f in picked]
    if not per_class:
        raise RuntimeError(
            "EXPL_CENTER=calibration: the concept map has no class"
        )
    path = config.explain_dir / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if config.single_object:
        images = [(p, cls) for cls, ps in per_class.items() for p in ps]
        data = {
            "images": [p for p, _ in images],
            "labels": [c for _, c in images],
        }
    else:
        classes = sorted(per_class)
        n_cells = 4 if len(classes) >= 4 else 1
        n_grids = max(
            1, len(classes) * config.expl_calibration_per_class // n_cells
        )
        grids = []
        for _ in range(n_grids):
            labels = rng.sample(classes, n_cells)
            grids.append(
                {
                    "cells": [rng.choice(per_class[c]) for c in labels],
                    "labels": labels,
                }
            )
        data = {"cell_size": 384, "grids": grids}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return path


def step_6_vlm_explainer(config: PipelineConfig, logger: logging.Logger):
    """Step 6: VLM explainer per method.

    Runs as a subprocess — loads VLM model for explanation generation.
    EXPL_MAX_IMAGES: -1 = all IMAGE_ROOT images, 0 = skip this step,
    N = first N images.
    """

    if config.expl_max_images == 0:
        logger.info(
            "Skip Step 6 (EXPL_MAX_IMAGES=0 — explanation disabled); "
            "steps 7-8 will have no input."
        )
        return

    step_5_recomputed = False  # Track if any method was re-computed

    for method in _active_methods(config, logger):
        method = method.strip()
        concept_path = (
            config.decomp_dir
            / method
            / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        )
        out_dir = config.explain_dir / method
        out_json = out_dir / "vlm_explanations.json"

        if out_json.exists():
            logger.info(f"Skip Explainer ({method}) (found {out_json})")
            continue

        # Mark that step 5 is being re-computed for this method
        step_5_recomputed = True

        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"START: Explainer ({method})")

        explainer_args = [
            "-m",
            "tgcl.inference.vlm_explainer_multibatch",
            "--model_name",
            config.vlm_model,
            "--concept_path",
            str(concept_path),
            "--layer_path",
            config.layer_path,
            "--image_root",
            str(config.image_root),
            "--top_n",
            str(config.top_n),
            "--out_json",
            str(out_json),
            "--prompt_mode",
            config.expl_prompt_mode,
            "--max_images",
            str(config.expl_max_images),
            "--batch_size",
            str(config.expl_batch_size),
        ]

        if config.expl_label:
            explainer_args.extend(["--prompt_label", config.expl_label])
        if config.expl_choices_file is not None:
            explainer_args.extend(
                ["--choices_file", str(config.expl_choices_file)]
            )
        elif config.expl_choices:
            explainer_args.extend(["--choices", config.expl_choices])
        if config.expl_mcq_distractors > 0:
            explainer_args.extend(
                ["--mcq_distractors", str(config.expl_mcq_distractors)]
            )
        if config.single_object:
            explainer_args.append("--single_object")
        if config.expl_input_mode != "image":
            explainer_args.extend(["--input_mode", config.expl_input_mode])
        if config.expl_token_state != "predicting":
            explainer_args.extend(["--token_state", config.expl_token_state])
        if config.expl_center == "calibration":
            explainer_args.extend(
                ["--center_calibration", str(_calibration_list(config, logger))]
            )

        # Allocations above 128 KB go straight to mmap and back to the OS when
        # freed:
        # without this, glibc raises the threshold after large frees and the
        # explainer's heap fragments, growing by ~1.5 MB per image (≈75 GB at
        # 50k).
        _run_python_subprocess(
            explainer_args,
            env_overrides={"MALLOC_MMAP_THRESHOLD_": "131072"},
            logger=logger,
        )

        logger.info(f"DONE: Explainer ({method})")

    # If any method was re-computed, cascade delete downstream outputs
    if step_5_recomputed:
        _delete_downstream_outputs(config, 6, logger)


def _run_concept_curves(
    config: PipelineConfig,
    method: str,
    in_json: Path,
    concept_path: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> bool:
    """Concept insertion/deletion curves of one method's explanations
    (tgcl.evaluation.concept_insertion_deletion), in a subprocess; kept when
    concept_curves.json exists. Returns True if they were computed."""
    if (out_dir / CONCEPT_CURVES_SUMMARY).exists():
        logger.info(f"Skip Concept curves ({method}): found summary")
        return False
    logger.info(f"START: Concept curves ({method})")
    _run_python_subprocess(
        [
            "-m",
            "tgcl.evaluation.concept_insertion_deletion",
            "--results_json",
            str(in_json),
            "--concept_path",
            str(concept_path),
            "--model_name",
            config.vlm_model,
            "--layer_path",
            config.layer_path,
            "--out_dir",
            str(out_dir),
            "--device",
            config.device,
            "--seed",
            str(config.seed),
        ],
        logger=logger,
    )
    logger.info(f"DONE: Concept curves ({method})")
    return True


def step_7_evaluate(config: PipelineConfig, logger: logging.Logger):
    """Step 7: evaluation per method: concept insertion/deletion curves,
    CLIP/BERT of the retrieved concepts, concept-word alignment (each a
    subprocess)."""

    step_6_recomputed = False  # Track if any eval was re-computed

    for method in _active_methods(config, logger):
        method = method.strip()
        concept_path = (
            config.decomp_dir
            / method
            / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        )
        in_json = config.explain_dir / method / "vlm_explanations.json"
        out_dir = config.eval_dir / method

        if not in_json.exists():
            logger.info(
                f"Skip Eval ({method}) — no explanations at {in_json} "
                "(step 6 skipped via EXPL_MAX_IMAGES=0 or produced nothing)."
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        # Concept insertion/deletion curves (every input mode: they edit the
        # stored answer states, no image is re-read).
        if _run_concept_curves(
            config, method, in_json, concept_path, out_dir, logger
        ):
            step_6_recomputed = True

        # Top-k semantic table (BERT/CLIP) + random activation baseline
        logger.info(f"START: Eval BERT/CLIP Top-K Table ({method})")
        clip_bert_args = [
            "-m",
            "tgcl.evaluation.clip_bert_score_eval",
            "--json_path",
            str(in_json),
            "--concept_path",
            str(concept_path),
            "--max_k",
            str(config.top_n),
            "--seed",
            str(config.seed),
            "--out_dir",
            str(out_dir),
            "--output_prefix",
            "clip_bert_topk",
            "--grounding_image_width",
            str(config.image_size_width),
        ]
        # The binary prompt's "no"/"yes" answer words name no concept.
        if config.expl_prompt_mode == "binary":
            clip_bert_args.extend(["--ignore_words", "no,yes"])
        _run_python_subprocess(clip_bert_args, logger=logger)
        logger.info(f"DONE: Eval BERT/CLIP Top-K Table ({method})")

        # Concept-word alignment (test image vs grounded words, Rnd-Words
        # baseline).
        word_alignment_table = out_dir / "word_alignment_table.csv"
        if (
            config.expl_input_mode == "image"
            and not word_alignment_table.exists()
        ):
            logger.info(f"START: Eval concept-word alignment ({method})")
            align_args = [
                "-m",
                "tgcl.evaluation.word_alignment_eval",
                "--json_path",
                str(in_json),
                "--concept_path",
                str(concept_path),
                "--out_dir",
                str(out_dir),
                "--model_name",
                config.vlm_model,
                "--max_k",
                str(config.top_n),
                "--dl_alpha",
                str(config.dl_alpha),
                "--seed",
                str(config.seed),
                "--device",
                config.device,
            ]
            if config.expl_prompt_mode == "binary":
                align_args.extend(["--ignore_words", "no,yes"])
            # Captions alone give the merged-words BERTScore; phrases add the
            # LLaMA-3 one.
            if (
                config.word_align_captions_json is not None
                and config.word_align_captions_json.exists()
            ):
                align_args.extend(
                    ["--captions_json", str(config.word_align_captions_json)]
                )
                if (
                    config.word_align_phrases_json is not None
                    and config.word_align_phrases_json.exists()
                ):
                    align_args.extend(
                        ["--phrases_json", str(config.word_align_phrases_json)]
                    )
            _run_python_subprocess(align_args, logger=logger)
            logger.info(f"DONE: Eval concept-word alignment ({method})")

    # If any eval was re-computed, cascade delete downstream outputs (plots)
    if step_6_recomputed:
        _delete_downstream_outputs(config, 7, logger)


def step_8_plots(config: PipelineConfig, logger: logging.Logger):
    """Step 8: concept-curve plots per method, then the gaps across
    methods."""
    methods = _active_methods(config, logger)
    for method in methods:
        out = plot_method(config.eval_dir / method, title=method)
        if out is None:
            logger.warning(f"No concept curves for {method}; no plot")
        else:
            logger.info(f"Plot ({method}): {out}")
    out = plot_summary(
        config.eval_dir, methods, config.plots_dir / "summary_comparison.png"
    )
    if out is not None:
        logger.info(f"Plot summary: {out}")


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XL-VLMs Pipeline - Native Python Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        help="Root dataset/images directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Root output directory",
    )
    parser.add_argument(
        "--decomp",
        type=str,
        help="Comma-separated decomposition methods",
    )
    parser.add_argument(
        "--skip-to-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Skip to a specific step (for resuming)",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Stop after this step (e.g. 5 for concept shards)",
    )
    parser.add_argument(
        "--only-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Run only a specific step",
    )

    return parser.parse_args()


def main():
    """Main pipeline entry point."""
    # Parse CLI args
    args = parse_args()

    # Load configuration
    config = PipelineConfig()
    config.update_from_args(args)

    # Set environment variables
    os.environ["HF_HOME"] = str(config.hf_home)
    # Do NOT override CUDA_VISIBLE_DEVICES — device placement is handled by
    # device_utils.parse_device_config() using the DEVICE env var. Export the
    # resolved value (incl. the DEVICE_ID fallback) so every subprocess —
    # also the ones that take no --device flag — uses the same GPU(s).
    os.environ["DEVICE"] = config.device
    os.environ["DETECTION_BATCH_SIZE"] = str(config.detection_batch_size)
    os.environ["POSITIVE_NEGATIVE_SEGMENT"] = str(
        config.positive_negative_segment
    )
    os.environ["MASK_CONTEXT_PIXELS"] = str(config.mask_context_pixels)
    os.environ["OUTPUT_DIR"] = str(config.output_dir)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if os.environ.get("DETERMINISTIC", "0") == "1":
        # cuBLAS reads this when CUDA starts: set it for every step's process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Create directories and setup logging
    create_directories(config, logging.getLogger())
    logger = setup_logging(config.logs_dir)

    # Log configuration
    logger.info("=" * 60)
    logger.info("XL-VLMs Pipeline Started")
    logger.info("=" * 60)
    logger.info("Root:       %s", config.root_dir)
    logger.info("Input dir:  %s", config.input_dir)
    logger.info("Output dir: %s", config.output_dir)
    logger.info(
        "Model:      %s | Batch: %s (step 1: %s) | Seed: %s | Device: %s",
        config.vlm_model,
        config.batch_size,
        config.inference_batch_size,
        config.seed,
        config.device,
    )
    logger.info("Decompose:  %s", ", ".join(config.decomp_methods))
    logger.info(
        "Crops: detector=%s masks=%s concept=%s patch=%s min=%s max=%s",
        config.object_detector,
        config.masks_per_image,
        config.concept_masks_per_image,
        config.patch_size,
        config.min_images_per_tag,
        config.max_images_per_tag,
    )
    logger.info(
        "Masking: ctx=%s pos_neg_segment=%s",
        config.mask_context_pixels,
        config.positive_negative_segment,
    )
    logger.info(
        "Resize:     ref_width=%s (height auto by aspect ratio)",
        config.image_size_width,
    )
    logger.info(
        "Explainer:  layer=%s image_root=%s top_n=%s mode=%s",
        config.layer_path,
        config.image_root,
        config.top_n,
        config.expl_prompt_mode,
    )
    logger.info("Hooks:      %s", config.hook_names)
    if config.logit_lens_layer_selection:
        logger.info(
            "LayerSelect: ON mode=%s layers=%s regions=%s (from crops.json)",
            config.logit_lens_mode,
            config.logit_lens_layers,
            config.logit_lens_num_patches,
        )
    logger.info(
        "Grounding:  num_most_activating_samples=%s",
        config.num_most_activating_samples,
    )
    logger.info("=" * 60)

    # Define pipeline steps
    steps = [
        (1, "Dataset Inference", step_1_dataset_inference),
        (2, "Build Crops JSON", step_2_build_crops_json),
        (3, "Select Layers (Logit Lens)", step_3_select_layers),
        (4, "Generate Features", step_4_generate_features),
        (5, "Decompose Features", step_5_decompose_features),
        (6, "VLM Explainer", step_6_vlm_explainer),
        (7, "Evaluation", step_7_evaluate),
        (8, "Plots", step_8_plots),
    ]

    # Determine which steps to run
    start_step = args.skip_to_step if args.skip_to_step else 1
    only_step = args.only_step

    try:
        for step_num, step_name, step_func in steps:
            if only_step is not None and step_num != only_step:
                continue
            if step_num < start_step:
                logger.info("Skipping Step %s: %s", step_num, step_name)
                continue
            if (
                args.stop_after_step is not None
                and step_num > args.stop_after_step
            ):
                break

            logger.info("Running Step %s: %s", step_num, step_name)
            step_func(config, logger)

        logger.info("=" * 60)
        logger.info("Pipeline completed. Outputs: %s", config.output_dir)
        logger.info("Logs: %s", config.logs_dir)
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
