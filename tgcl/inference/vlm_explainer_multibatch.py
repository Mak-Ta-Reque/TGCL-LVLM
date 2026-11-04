#!/usr/bin/env python3
"""Vision-Language Model Concept Explainer (Gemma 3n)

- Loads Gemma 3n model & processor
- Hooks a target layer to capture activations
- Compares pooled activations with concept vectors via cosine similarity
- Returns model output + top-N concept matches with grounding

Memory-aware:
- Resizes images to max 512 on the longer side
- Moves captured activations to CPU immediately
- Keeps concept vectors on CPU and computes similarities on CPU
- Supports small batch inference; default batch_size=1
"""

from __future__ import annotations

import gc
import os

# Suppress noisy third-party logs early
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # hide TF INFO/WARNING/ERROR
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
# The variables above must be set before transformers is imported.
# pylint: disable=wrong-import-position
import argparse
import json
import logging
from pathlib import Path
import pickle
import random
import re
import time as _time
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers.utils import logging as hf_logging

from tgcl.device_utils import get_device_config
from tgcl.helpers import image_refs
from tgcl.helpers import utils as hook_utils
from tgcl.helpers.batching import collate_inputs
from tgcl.helpers.explanations import compact
from tgcl.helpers.utils import set_seed
from tgcl.models import get_model_class
from tgcl.models.constants import TASK_PROMPTS

# pylint: enable=wrong-import-position

# Route HF logs through Python logging and keep them quiet by default
hf_logging.set_verbosity_error()
hf_logging.disable_default_handler()
hf_logging.enable_propagation()
# Silence common deprecation/runtime warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='transformers')

# Note: Specific model classes are loaded via the project's model loader.


INPUT_MODES = ('image', 'text_only')
# Template whose "Repeat the text" prompt drives text-only input.
TEXT_ONLY_PROMPT_TEMPLATE = 'cgdl'


def text_only_instruction(label: str) -> str:
    """Instruction that makes the model write `label` (underscores: spaces)."""
    word = (label or '').replace('_', ' ').strip()
    if not word:
        raise ValueError('text_only input needs a class label for every sample')
    return (
        f"{TASK_PROMPTS[TEXT_ONLY_PROMPT_TEMPLATE]['Repeat the text']}: "
        f"'{word}'"
    )


_POOL_SKIP = re.compile(
    r'\b(?:a|an|the)\b|\b(?:'
    r'i|object|you|he|she|it|we|they|me|him|her|us|them|my|mine|your|yours|'
    r'his|hers|its|our|ours|that|their|theirs|myself|yourself|himself|'
    r'herself|itself|ourselves|yourselves|themselves|'
    r'photo|image|im_end|question)\b',
    flags=re.IGNORECASE,
)


def keep_token_text(s: str) -> bool:
    """Whether a generated token enters the sequence-level (pooled) concept
    ranking:
    it has a letter or digit left after removing articles, pronouns and filler
    words.
    """
    if not s or not any(ch.isalnum() for ch in s):
        return False
    txt = re.sub(r'[^a-zA-Z0-9\s]', '', _POOL_SKIP.sub('', s))
    return re.sub(r'\s+', ' ', txt).strip() != ''


def cosine_to_concepts(
    x: torch.Tensor, unit_concepts: torch.Tensor
) -> torch.Tensor:
    """Cosine similarity of every row of x (T, D) with every unit-norm concept
    (K, D),
    as one matmul: the values of F.cosine_similarity(x[:, None], concepts[None],
    dim=2)
    without its T x K x D float temporary (16 MB per token for a 2000-concept
    bank,
    which fragments the heap over 50k images)."""
    return F.normalize(x.to(torch.float32), dim=-1, eps=1e-8) @ unit_concepts.T


def resolve_prompt_labels(
    image_paths: List[Optional[str]],
    ground_truth_labels: Optional[List[Optional[str]]] = None,
    prompt_label: Optional[str] = None,
) -> List[Optional[str]]:
    """Class label per sample for prompts that name the class.

    Precedence: an explicit prompt_label, then the given ground-truth labels,
    then the image's class (its class folder; helpers.image_refs.image_classes).
    """
    if prompt_label is not None:
        return [prompt_label] * len(image_paths)
    if ground_truth_labels:
        return list(ground_truth_labels)
    return [(image_classes(p) or [None])[0] if p else None for p in image_paths]


def label_start_indices(token_texts: List[str]) -> List[int]:
    """Indices of the tokens that start a label in an answer: the first word
    token, and the first word
    token after each comma ("acorn, school bus, ..." -> "acorn", "school").
    These are the tokens whose
    states decide the class (Obj-F1 reads the first one), so the calibration
    mean is taken over them:
    averaging every word token (with "bus" of a split "school bus") shifted the
    mean and brought the hub
    back for Qwen2.5-VL (top-1 F1 0.996 -> 0.29 on ImageNet-5)."""
    out, start = [], True
    for i, s in enumerate(token_texts):
        if ',' in s:
            start = True
            continue
        if start and keep_token_text(s):
            out.append(i)
            start = False
    return out


def states_at_tokens(captured: torch.Tensor, n_generated: int) -> torch.Tensor:
    """(n_generated, D) states AT each generated token, from a cached
    generation's captured states (positions 0 .. L+T-2: the prompt, then one per
    decoding step; the last token is never fed back). The state at token k is
    position L+k, the row after the one predicting it; the last token has no own
    state and keeps its predicting state."""
    if n_generated <= 1:
        return captured[-n_generated:]
    return torch.cat([captured[-(n_generated - 1) :], captured[-1:]], dim=0)


def image_classes(image_path: Optional[str]) -> List[str]:
    """True class name(s) of an evaluation image reference ('_' -> ' '): see
    helpers.image_refs."""
    return image_refs.image_classes(image_path)


def sample_mcq_choices(
    image_path: Optional[str],
    pool: List[str],
    n_distractors: int,
    seed: int,
    true_classes: Optional[List[str]] = None,
) -> List[str]:
    """The image's true class(es) (true_classes, else from its reference) plus
    n_distractors other classes from pool, shuffled. Deterministic per (seed,
    image file name)."""

    def norm(c):
        return c.replace('_', ' ').strip().lower()

    true = list(true_classes) if true_classes else image_classes(image_path)
    true_norm = {norm(c) for c in true}
    pool_by_norm = {norm(c): c for c in pool}
    missing = [c for c in true if norm(c) not in pool_by_norm]
    if missing:
        raise ValueError(
            f'{image_path}: classes {missing} are not among the '
            f'{len(pool)} choices'
        )
    rng = random.Random(f'{seed}:{image_refs.ref_name(image_path)}')
    others = sorted(c for n, c in pool_by_norm.items() if n not in true_norm)
    choices = [pool_by_norm[norm(c)] for c in true] + rng.sample(
        others, min(n_distractors, len(others))
    )
    rng.shuffle(choices)
    return choices


class VLMConceptExplainer:
    """Explain a VLM's answers on test images with a concept bank (step 6).

    The model answers each test image (or, in text_only mode, writes its class
    name); the hidden state of every answer token at the hooked layer is
    scored against the concepts, and the top concepts per token form the
    explanation (with their groundings). With centring (set_centering) the
    scores use the states minus a calibration mean.

    Args:
        model_name: The VLM.
        concept_path: The concept bank (.pth with 'concepts').
        layer_path: The hooked layer (same as for the concepts).
        device: Device spec (default: DEVICE).
        trust_remote_code: Passed to the model loader.
        hf_token: Hugging Face token (default: HF_TOKEN).
        activation_pool: How token states are pooled ('mean' or a function).
        default_top_n: Concepts reported per token.
        normalize_concepts: Unit-normalize the concepts before scoring.
        capture_only_last: Keep only the last captured state per step.
        exact_match_modules_to_hook: layer_path is an exact module name.
        save_only_generated_tokens: Keep only the answer's states.
        verbose: Print progress.
        prompt_mode: unsupervised (open question), binary ("<label>" or "no
            <label>"), or mcq (choose from prompt_choices).
        prompt_label: The label for binary mode (default: the image's).
        prompt_choices: Class names for mcq (list or comma-separated).
        mcq_distractors: With mcq, ask about the true class(es) plus this many
            random other classes.
        mcq_seed: Seed of the distractor sampling.
        prompt: An explicit prompt (overrides prompt_mode).
        single_object: Test images are single objects, not grids of cells.
        input_mode: 'image', or 'text_only' (the text-only retrieval baseline).
        token_state: 'predicting' (the state that predicts each token) or
            'at' (the state at the token).

    Raises:
        ValueError: An invalid token_state, input_mode or mcq set-up.
    """

    REQUIRED_CONCEPT_KEYS = {'concepts'}

    def __init__(
        self,
        model_name: str,
        concept_path: Union[str, Path],
        layer_path: str,
        device: Optional[Union[str, torch.device]] = None,
        trust_remote_code: bool = True,
        hf_token: Optional[str] = None,
        activation_pool: Union[
            str, Callable[[torch.Tensor], torch.Tensor]
        ] = 'mean',
        default_top_n: int = 5,
        normalize_concepts: bool = False,
        capture_only_last: bool = True,
        exact_match_modules_to_hook: bool = True,
        save_only_generated_tokens: bool = False,
        verbose: bool = False,
        prompt_mode: str = 'unsupervised',
        prompt_label: Optional[str] = None,
        prompt_choices: Optional[Union[str, List[str]]] = None,
        mcq_distractors: int = 0,
        mcq_seed: int = 42,
        prompt: Optional[str] = None,
        single_object: bool = False,
        input_mode: str = 'image',
        token_state: str = 'predicting',
    ) -> None:
        if token_state not in ('predicting', 'at'):
            raise ValueError(
                f"token_state must be 'predicting' or 'at', got {token_state!r}"
            )
        self.token_state = token_state
        self.model_name = model_name
        self.layer_path = layer_path
        self.trust_remote_code = trust_remote_code
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.activation_pool = activation_pool
        self.default_top_n = default_top_n
        self.normalize_concepts = normalize_concepts
        self.capture_only_last = capture_only_last
        self.exact_match_modules_to_hook = exact_match_modules_to_hook
        self.verbose = verbose
        self.save_only_generated_tokens = save_only_generated_tokens
        # Prompt configuration
        self.prompt_mode = (prompt_mode or 'unsupervised').lower()
        self.prompt_label = prompt_label
        if isinstance(prompt_choices, str):
            # Accept comma-separated string
            self.prompt_choices = [
                c.strip() for c in prompt_choices.split(',') if c.strip()
            ]
        else:
            self.prompt_choices = prompt_choices
        # mcq with mcq_distractors > 0: each image is asked about its true
        # class(es)
        # plus that many random other classes from prompt_choices
        # (sample_mcq_choices).
        self.mcq_distractors = int(mcq_distractors)
        self.mcq_seed = int(mcq_seed)
        if self.mcq_distractors > 0 and not self.prompt_choices:
            raise ValueError(
                'mcq_distractors needs the class pool '
                '(--choices/--choices_file)'
            )
        self.prompt = prompt
        # Grid-cell wording ("this image is a grid of separate cells...")
        # breaks down on single-object images -- the model has no cells to
        # enumerate and either repeats one word or hallucinates several
        # different ones trying to satisfy "for each cell". Set this when
        # image_root points at single-object crops (not multi-cell grids).
        self.single_object = bool(single_object)
        # "image": the evaluation image + the prompt above. "text_only": no
        # image; the model is asked to write the image's class name
        # (TEXT_ONLY_PROMPT_TEMPLATE's "Repeat the text" prompt), which gives
        # the text-only concept-retrieval baseline.
        if input_mode not in INPUT_MODES:
            raise ValueError(
                f'input_mode must be one of {INPUT_MODES}, got {input_mode!r}'
            )
        self.input_mode = input_mode

        # Memory behavior
        os.environ.setdefault(
            'PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True'
        )

        # Load model and processor via repo's loader
        self.model, self.processor = self._load_model(device)
        self.device = (
            getattr(self, '_device_config', None)
            and self._device_config.primary_device
            or (
                torch.device(device)
                if device
                else next(self.model.parameters()).device
            )
        )

        # Load concept vectors and metadata
        self.concept_data = self._load_concepts(concept_path)
        # Support either 'concepts' or legacy 'activations' key
        vec_key = (
            'concepts'
            if 'concepts' in self.concept_data
            else ('activations' if 'activations' in self.concept_data else None)
        )
        if vec_key is None:
            raise KeyError(
                "Concept file must contain 'concepts' or 'activations'."
            )
        self.concept_vectors = self._prepare_concept_vectors(
            self.concept_data[vec_key]
        )  # (K, D), CPU
        self.num_concepts, self.embed_dim = self.concept_vectors.shape
        self._unit_concepts = F.normalize(
            self.concept_vectors, dim=1, eps=1e-8
        )  # for cosine_to_concepts
        self._answer_center: Optional[torch.Tensor] = None  # set_centering
        self.concept_names = self.concept_data.get(
            'concept_names'
        ) or self.concept_data.get('names')
        self.text_grounding = self.concept_data.get('text_grounding')
        self.image_grounding_paths = self.concept_data.get(
            'image_grounding_paths'
        )
        self.image_grounding_bboxes = self.concept_data.get(
            'image_grounding_bboxes'
        )
        self.image_grounding_masks = self.concept_data.get(
            'image_grounding_masks'
        )

        # Hook setup via project utils
        self._hook_return_fn = None
        self._hook_post_fn = None
        self._setup_utils_hooks()

    def set_centering(self, answer_center: torch.Tensor) -> None:
        """Centred concept matching. Concepts and answer states each share a
        large component (the concepts come from forced answers "There is a
        <class>.", the states from the explainer's own answers), so raw cosine
        makes one concept the nearest to almost every answer: a hub (COCO,
        Gemma-3n: one concept rank 1 for 480 of 500 images). Matching subtracts
        the bank mean from the concepts and `answer_center` -- the mean matching
        state of calibration answers on training images (calibration_states) --
        from every answer state. Stored activations stay raw.
        """
        self._answer_center = (
            answer_center.detach().float().cpu().reshape(1, -1)
        )
        vectors = self.concept_vectors.float()
        self._unit_concepts = F.normalize(
            vectors - vectors.mean(dim=0, keepdim=True), dim=1, eps=1e-8
        )

    def calibration_states(
        self,
        images: List[Any],
        labels: List[Optional[str]],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """(n, D) matching states of the answer words over calibration images,
        under the same prompt, input mode and token state as the explained
        images."""
        states: List[torch.Tensor] = []
        self.explain_with_concept(
            images,
            ground_truth_labels=labels,
            batch_size=batch_size,
            collect_states=states,
        )
        if not states:
            raise RuntimeError('calibration produced no answer word state')
        return torch.cat(states, dim=0)

    def _centred(self, states: torch.Tensor) -> torch.Tensor:
        """Answer states as matched to the concepts (minus the calibration mean
        when centring)."""
        if self._answer_center is None:
            return states
        return states.float().cpu() - self._answer_center

    # ---------- Concept IO ----------
    def _load_concepts(self, path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Concept file not found: {p}')
        if p.suffix.lower() == '.json':
            return json.loads(p.read_text(encoding='utf-8'))
        if p.suffix.lower() in {'.pkl', '.pickle'}:
            with p.open('rb') as f:
                return pickle.load(f)
        if p.suffix.lower() in {'.pt', '.pth'}:
            return torch.load(str(p), map_location='cpu')
        if p.suffix.lower() == '.npz':
            npz = np.load(str(p), allow_pickle=True)
            return {
                k: (npz[k].tolist() if hasattr(npz[k], 'tolist') else npz[k])
                for k in npz.files
            }
        raise ValueError(f'Unsupported concept file extension: {p.suffix}')

    def _prepare_concept_vectors(self, raw_vectors: Any) -> torch.Tensor:
        if isinstance(raw_vectors, torch.Tensor):
            vecs = raw_vectors.float()
        else:
            vecs = torch.tensor(np.asarray(raw_vectors), dtype=torch.float32)
        if vecs.dim() != 2:
            raise ValueError(
                f'Concept activation matrix must be 2D, got {tuple(vecs.shape)}'
            )
        if self.normalize_concepts:
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        return vecs.to('cpu')

    # ---------- Model ----------
    def _load_model(
        self, device_override: Optional[Union[str, torch.device]] = None
    ):
        """Load model via the project's model loader and use its preprocessor.

        Falls back to HF Auto* only if import fails (shouldn't happen in this
        repo).
        """
        device_config = get_device_config(
            str(device_override) if device_override is not None else None
        )
        device = device_config.primary_device
        self._device_config = device_config

        # Minimal args namespace expected by get_model_class
        args = argparse.Namespace(
            local_files_only=False,
            cache_dir=None,  # default HF cache ($HF_HOME/hub)
        )

        model_class = get_model_class(
            model_name_or_path=self.model_name,
            processor_name=self.model_name,
            device=device,
            logger=None,
            args=args,
            device_config=device_config,
        )
        self._model_class = model_class
        model = model_class.get_model()
        processor = model_class.get_processor()
        return model, processor

    # ---------- Hook ----------
    def _parse_layer_patterns(self) -> List[str]:
        raw = self.layer_path or ''
        parts: List[str] = []
        for sep in (',', '|'):
            if sep in raw:
                parts.extend([p.strip() for p in raw.split(sep)])
                break
        if not parts:
            parts = [raw.strip()]
        # de-dup and drop empties
        uniq: List[str] = []
        for p in parts:
            if p and p not in uniq:
                uniq.append(p)
        return uniq

    def _setup_utils_hooks(self) -> None:
        """Register the hidden-state hook on layer_path (previous hooks and
        buffers are cleared)."""
        hook_utils.clear_forward_hooks(self.model)
        hook_utils.clear_hooks_variables()

        modules_to_hook = [self._parse_layer_patterns()]
        hook_names = ['save_hidden_states']
        # Minimal args namespace controlling exact match
        hook_args = argparse.Namespace(
            exact_match_modules_to_hook=self.exact_match_modules_to_hook,
            verbose=self.verbose,
            save_only_generated_tokens=self.save_only_generated_tokens,
        )
        hook_ret, hook_post = hook_utils.setup_hooks(
            model=self.model,
            modules_to_hook=modules_to_hook,
            hook_names=hook_names,
            tokenizer=getattr(
                self._model_class, 'get_tokenizer', lambda: None
            )(),
            logger=None,
            args=hook_args,
        )
        self._hook_return_fn = hook_ret[0] if hook_ret else None
        self._hook_post_fn = hook_post[0] if hook_post else None

    # ---------- Prompts ----------
    def _choices_for(
        self, image_path: Optional[str], label: Optional[str] = None
    ) -> Optional[List[str]]:
        """Choices shown for one image: sampled ones (mcq_distractors), else the
        full list. `label`: the image's class when its path does not tell it
        (calibration images from a training folder).
        """
        if (self.prompt_mode or '').lower() != 'mcq':
            return None
        if self.mcq_distractors > 0:
            return sample_mcq_choices(
                image_path,
                self.prompt_choices,
                self.mcq_distractors,
                self.mcq_seed,
                true_classes=[label] if label else None,
            )
        return self.prompt_choices

    def _build_instruction(
        self, label: Optional[str], choices: Optional[List[str]] = None
    ) -> str:
        # If custom prompt is provided, use it (especially for unsupervised
        # mode)
        if self.prompt is not None:
            return self.prompt

        mode = (self.prompt_mode or 'unsupervised').lower()
        if mode == 'binary':
            lab = self.prompt_label or label
            if lab:
                return (
                    f"Binary classification: Say  '{lab}' if the image "
                    f"contains '{lab}', say 'no {lab}' if it does not. "
                    f'Respond with only "{lab}" or "no {lab}".'
                )
            # Fallback to unsupervised if no label provided
        if mode == 'mcq':
            choices = (
                choices if choices is not None else (self.prompt_choices or [])
            )
            if choices:
                choice_str = ', '.join(choices)
                if self.single_object:
                    return (
                        'This image shows a single object. Choose the '
                        'single best-matching '
                        f'label from this list: {choice_str}. '
                        'Respond with exactly one label from that list — '
                        'no other words, '
                        'no repetition, no punctuation.'
                    )
                return (
                    'This image is a grid of separate cells, each showing '
                    'one object. '
                    'For each cell, in reading order (left-to-right, '
                    'top-to-bottom), choose the single '
                    f'best-matching label from this list: {choice_str}. '
                    'Output exactly one label per cell, in a single line, '
                    'separated by commas, '
                    'using only labels from that list — no other words.'
                )
            # Fallback to unsupervised if no choices provided
        # Unsupervised/default prompt
        if self.single_object:
            return 'Name the main object in this image in a single word. '
        return (
            'Name the main object in each grid in a single line, separated by '
            'commas. '
        )

    # ---------- Utils ----------

    @staticmethod
    def _resize_if_large(img: Image.Image, max_side: int = 512) -> Image.Image:
        w, h = img.size
        m = max(w, h)
        if m <= max_side:
            return img
        scale = max_side / float(m)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return img.resize((nw, nh), Image.Resampling.LANCZOS)

    # Note: pooling helpers removed; per-token and mean-over-selected tokens are
    # applied inline.

    # ---------- API ----------
    @torch.inference_mode()
    def explain_with_concept(
        self,
        images: List[Union[str, Path, Image.Image]],
        ground_truth_labels: Optional[List[Optional[str]]] = None,
        top_n: Optional[int] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.0,
        batch_size: int = 1,
        save_only_generated_tokens: bool = False,
        collect_states: Optional[List[torch.Tensor]] = None,
    ) -> List[Dict[str, Any]]:
        """collect_states: calibration -- append each answer's matching states
        of its word tokens to this list instead of ranking concepts (no results
        are returned)."""
        if ground_truth_labels and len(ground_truth_labels) != len(images):
            raise ValueError(
                'ground_truth_labels length must match images length'
            )
        N = top_n if top_n is not None else self.default_top_n
        N = min(N, getattr(self, 'num_concepts', N))
        # Preload + resize PIL if needed, but keep original input (path or PIL)
        # for the model preprocessor
        prepped_inputs: List[Union[Image.Image, str, Path]] = []
        abs_image_paths: List[Optional[str]] = []
        for img_in in images:
            if image_refs.is_memory_ref(img_in):
                # A grid or crop is built in memory from its spec; the reference
                # stays its name.
                prepped_inputs.append(
                    self._resize_if_large(image_refs.load_image(img_in))
                )
                abs_image_paths.append(img_in)
            elif isinstance(img_in, (str, Path)):
                # Keep path; let model preprocessor load/resize
                prepped_inputs.append(str(img_in))
                abs_image_paths.append(os.path.abspath(str(img_in)))
            elif isinstance(img_in, Image.Image):
                prepped_inputs.append(
                    self._resize_if_large(img_in.convert('RGB'))
                )
                abs_image_paths.append(None)
            else:
                raise TypeError(f'Unsupported image type: {type(img_in)}')

        # Labels used in the instruction (not ground truth). Prompts that name
        # the class (binary mode, text-only input) need one per sample.
        text_only = self.input_mode == 'text_only'
        if text_only or (self.prompt_mode or '').lower() == 'binary':
            labels_all = resolve_prompt_labels(
                abs_image_paths, ground_truth_labels, self.prompt_label
            )
        else:
            labels_all = (
                ground_truth_labels
                if ground_truth_labels
                else [None] * len(prepped_inputs)
            )

        results: List[Dict[str, Any]] = []

        def chunks(seq, n):
            for i in range(0, len(seq), n):
                yield i, seq[i : i + n]

        total_images = len(prepped_inputs)
        progress_start = _time.time()
        logging.info(
            '[explainer] Processing %s images (batch_size=%s, '
            'max_new_tokens=%s)',
            total_images,
            batch_size,
            max_new_tokens,
        )

        for base_idx, img_chunk in chunks(prepped_inputs, batch_size):
            lab_chunk = labels_all[base_idx : base_idx + len(img_chunk)]
            choices_chunk = [
                self._choices_for(
                    abs_image_paths[base_idx + i],
                    ground_truth_labels[base_idx + i]
                    if ground_truth_labels
                    else None,
                )
                for i in range(len(img_chunk))
            ]
            # Reset utils hook buffers
            hook_utils.clear_hooks_variables()

            # Build per-sample inputs using repo preprocessor
            per_sample_inputs: List[Dict[str, Any]] = []
            for i in range(len(img_chunk)):
                per_sample_inputs.append(
                    self._model_class.preprocessor(
                        instruction=(
                            text_only_instruction(lab_chunk[i])
                            if text_only
                            else self._build_instruction(
                                lab_chunk[i], choices_chunk[i]
                            )
                        ),
                        image_file='' if text_only else img_chunk[i],
                        response='',
                        generation_mode=True,
                    )
                )

            # Acquire tokenizer and pad id
            tokenizer = None
            if getattr(self, '_model_class', None) is not None:
                tokenizer = getattr(
                    self._model_class, 'get_tokenizer', lambda: None
                )()
            if tokenizer is None and hasattr(self.processor, 'tokenizer'):
                tokenizer = self.processor.tokenizer
            pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0

            # Left-pads input_ids and every token-aligned tensor
            # (attention_mask,
            # token_type_ids, ...) together; the same collation as step 4.
            inputs = collate_inputs(per_sample_inputs, pad_id=pad_id)
            # Move tensors to device
            inputs = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            # Inputs are LEFT-padded to a uniform width, so generated tokens
            # start
            # at this width for every row (decoder-only models).
            pre_len = int(inputs['input_ids'].shape[1])

            # Generation
            # Simple generation call without constructing a GenerationConfig
            out_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                pad_token_id=(pad_id if tokenizer is not None else None)
                or (
                    getattr(tokenizer, 'eos_token_id', None)
                    if tokenizer is not None
                    else None
                ),
                use_cache=True,
                early_stopping=True,
                # Gemma-3n's static cache makes transformers torch.compile the
                # forward; the hidden-state hook appends to a global list, which
                # forces a recompile per token and fails on answers > 8 tokens.
                disable_compile=True,
            )

            # Ensure 2D
            if out_tokens.ndim == 1:
                out_tokens = out_tokens.unsqueeze(0)

            # Slice generated token ids per sample
            # For decoder-only models, `generate` returns [input_ids |
            # new_tokens].
            # For encoder-decoder models, it usually returns only new tokens.
            is_enc_dec = bool(
                getattr(
                    getattr(self.model, 'config', None),
                    'is_encoder_decoder',
                    False,
                )
            )
            B = out_tokens.size(0)
            gen_token_ids: List[List[int]] = []
            if not is_enc_dec and out_tokens.size(1) >= pre_len:
                # Decoder-only path: new tokens start right after the
                # (left-padded) input width
                for b in range(B):
                    gen_b = out_tokens[b, pre_len:]
                    gen_token_ids.append(gen_b.tolist())
            else:
                # Encoder-decoder or unexpected shapes: treat the whole sequence
                # as generated
                for b in range(B):
                    gen_token_ids.append(out_tokens[b].tolist())

            # Decode per sample
            batch_texts: List[str] = []
            for ids in gen_token_ids:
                if tokenizer is not None:
                    batch_texts.append(
                        tokenizer.decode(ids, skip_special_tokens=True)
                    )
                else:
                    batch_texts.append(
                        self.processor.decode(ids, skip_special_tokens=True)
                    )

            if self.verbose:
                logging.debug('Generated tokens (debug):')
                for text in batch_texts:
                    logging.debug(' - %s', text)

            # Activation sequence via utils hook: hidden_states dict ->
            # per-sample tensors
            hidden_dict: Dict[str, List[torch.Tensor]] = {}
            if self._hook_return_fn is not None:
                # pylint: disable-next=not-callable  (checked for None above)
                hook_out = self._hook_return_fn(
                    tokenizer=tokenizer,
                    image=img_chunk,
                    model_output=out_tokens,  # for completeness
                    model_generated_output=gen_token_ids,
                )
                hidden_dict = (
                    hook_out.get('hidden_states', {})
                    if isinstance(hook_out, dict)
                    else {}
                )
                if len(hidden_dict) == 0:
                    raise RuntimeError('No hidden_states captured by hooks')

            # Pick the first hooked module deterministically
            first_key = sorted(hidden_dict.keys())[0]
            act_list: List[torch.Tensor] = hidden_dict[
                first_key
            ]  # list of (T, D) per sample

            # For each sample, compute per-token and pooled top-N concepts
            for j in range(len(act_list)):
                new_ids = gen_token_ids[j]
                T_new = len(new_ids)
                acts_j_full = act_list[j]
                T_cap = acts_j_full.shape[0]

                if T_cap < T_new:
                    raise RuntimeError(
                        f'Captured activation length {T_cap} shorter than '
                        f'generated tokens {T_new}; hook error?'
                    )
                # If hook did not save only generated, align to the last T_new
                # steps
                if not (
                    self.save_only_generated_tokens
                    or save_only_generated_tokens
                ):
                    acts_j = acts_j_full[-T_new:, :]
                else:
                    acts_j = acts_j_full[:T_new, :]
                # State matched to the concepts per token. "predicting"
                # (default): the state that
                # predicts the token, as stored (step 7 feeds it to the output
                # head). "at": the state
                # AT the token, like TGCL's forced-answer features
                # (FORCED_RESPONSE). With the KV cache
                # acts_j_full ends with the states of positions L-1 .. L+T-2, so
                # the state at token k
                # is the next row; the last token's own state is never computed
                # (it keeps its
                # predicting state; it is the end-of-turn token unless
                # max_new_tokens cut the answer).
                acts_match = acts_j
                if self.token_state == 'at':
                    if (
                        self.save_only_generated_tokens
                        or save_only_generated_tokens
                    ):
                        raise ValueError(
                            "token_state='at' needs the full captured "
                            'sequence (save_only_generated_tokens off)'
                        )
                    acts_match = states_at_tokens(acts_j_full, T_new)
                # Raw activations (T, D): cosine similarity below normalizes
                # each token vector itself, and the stored per-token
                # activations must be what the model computed
                # (concept_insertion_deletion feeds them to the output head).
                # Never normalize over the token axis (dim=0): that rescales
                # every feature by its norm across the sequence.
                if (
                    collect_states is not None
                ):  # calibration: the states of the tokens that start a label
                    dec = tokenizer or getattr(
                        self.processor, 'tokenizer', None
                    )
                    texts_j = [
                        dec.decode([t], skip_special_tokens=True)
                        if dec is not None
                        else str(t)
                        for t in new_ids
                    ]
                    keep = label_start_indices(texts_j)
                    if keep:
                        collect_states.append(acts_match[keep].float().cpu())
                    continue
                sims_tok = cosine_to_concepts(
                    self._centred(acts_match), self._unit_concepts
                )  # (T, K)
                dists_tok = 1.0 - sims_tok

                # Decode tokens one-by-one
                tok_decoder = tokenizer or getattr(
                    self.processor, 'tokenizer', None
                )
                token_texts: List[str] = []
                for tid in new_ids:
                    if tok_decoder is not None:
                        token_texts.append(
                            tok_decoder.decode([tid], skip_special_tokens=True)
                        )
                    else:
                        token_texts.append(str(tid))

                # Filter out empty/whitespace-only tokens from per-token output
                # (keep alphanumeric content only)
                def _is_nonempty_token(s: str) -> bool:
                    return bool(s) and any(ch.isalnum() for ch in s)

                kept_indices = [
                    i
                    for i, s in enumerate(token_texts)
                    if _is_nonempty_token(s)
                ]

                per_token_concepts: List[Dict[str, Any]] = []
                for t_idx in kept_indices:
                    dists = dists_tok[t_idx]
                    k = min(N, dists.shape[0])
                    topk = torch.topk(dists, k=k, largest=False)
                    concept_indices = topk.indices.tolist()
                    concept_scores = topk.values.tolist()
                    top_concepts_tok: List[Dict[str, Any]] = []
                    for rank, (ci, dist_val) in enumerate(
                        zip(concept_indices, concept_scores), 1
                    ):
                        top_concepts_tok.append(
                            {
                                'rank': rank,
                                'concept_index': ci,
                                'distance': float(dist_val),
                                'similarity': float(1.0 - dist_val),
                                **self._concept_fields(ci),
                            }
                        )
                    # Store token embedding temporarily for potential projection
                    # later (not returned in response)
                    # A copy: a view would keep the whole sequence's hidden
                    # states alive
                    # (when they are float32 already, .to() returns the same
                    # storage).
                    token_embedding = (
                        acts_j[t_idx]
                        .detach()
                        .to(dtype=torch.float32)
                        .cpu()
                        .numpy()
                        .copy()
                    )
                    per_token_concepts.append(
                        {
                            'token_index': t_idx,
                            'token_id': new_ids[t_idx],
                            'token_text': token_texts[t_idx],
                            # Internal use only, will be removed
                            '_token_embedding_np': token_embedding,
                            'top_concepts': top_concepts_tok,
                        }
                    )

                # Pooled over sequence (filter trivial tokens)
                mask_keep = [keep_token_text(t) for t in token_texts]
                if any(mask_keep):
                    acts_keep = acts_match[
                        torch.tensor(mask_keep, dtype=torch.bool)
                    ]
                else:
                    acts_keep = acts_match
                pooled = acts_keep.mean(dim=0, keepdim=True)  # (1, D)
                sims_all = cosine_to_concepts(
                    self._centred(pooled.to('cpu')), self._unit_concepts
                )[0]
                arr = sims_all.detach().cpu().numpy()
                k_all = min(N, arr.shape[0])
                idx_all = np.argsort(arr)[-k_all:][::-1].tolist()
                sim_vals = [float(arr[i]) for i in idx_all]
                top_concepts_all: List[Dict[str, Any]] = []
                for rank, (ci, sim_val) in enumerate(zip(idx_all, sim_vals), 1):
                    top_concepts_all.append(
                        {
                            'rank': rank,
                            'concept_index': ci,
                            'similarity': float(sim_val),
                            **self._concept_fields(ci),
                        }
                    )

                results.append(
                    {
                        'image_path': abs_image_paths[base_idx + j]
                        if (base_idx + j) < len(abs_image_paths)
                        else None,
                        'ground_truth': lab_chunk[j]
                        if j < len(lab_chunk)
                        else None,
                        'prompt_choices': choices_chunk[j],
                        'input_mode': self.input_mode,
                        'model_output': batch_texts[j],
                        'generated_token_ids': new_ids,
                        'per_token_concepts': per_token_concepts,
                        'top_concepts_over_sequence': top_concepts_all,
                        'hook_layer': self.layer_path,
                        'model_name': self.model_name,
                    }
                )

            done = min(base_idx + len(img_chunk), total_images)
            elapsed_value = _time.time() - progress_start
            rate = done / elapsed_value if elapsed_value > 0 else 0.0
            eta_min = (
                (total_images - done) / rate / 60 if rate > 0 else float('inf')
            )
            logging.info(
                '[explainer] %s/%s images | elapsed %.1fm | ETA %.1fm',
                done,
                total_images,
                elapsed_value / 60,
                eta_min,
            )

        return results

    def _concept_fields(self, ci: int) -> Dict[str, Any]:
        """Name, words and image grounding of concept ci (None where the bank
        has none). Tuples become lists; a single grounding path or box is
        stored as a string."""

        def at(values, to_str=False):
            if not values or ci >= len(values):
                return None
            val = values[ci]
            if isinstance(val, tuple):
                return list(val)
            if isinstance(val, list) or not to_str:
                return val
            return str(val)

        return {
            'concept_name': at(self.concept_names),
            'text_grounding': at(self.text_grounding),
            'image_grounding_path': at(self.image_grounding_paths, True),
            'image_grounding_bboxes': at(self.image_grounding_bboxes, True),
            'image_grounding_masks': at(self.image_grounding_masks),
        }

    def close(self):
        """Remove the hooks and clear their buffers."""
        hook_utils.clear_forward_hooks(self.model)
        hook_utils.clear_hooks_variables()


__all__ = ['VLMConceptExplainer']


def main():
    """Main entry point for vlm_explainer_multibatch script."""
    ap = argparse.ArgumentParser(
        description='Quick test for VLMConceptExplainer'
    )
    ap.add_argument('--model_name', default='google/gemma-3n-E4B-it')
    ap.add_argument('--concept_path', required=True)
    ap.add_argument('--layer_path', required=True)
    ap.add_argument(
        '--image', action='append'
    )  # removed required=True to allow --image_root only
    ap.add_argument(
        '--image_root',
        default=None,
        help='Directory to collect images from, or a JSON image / grid list '
        '(helpers.image_refs)',
    )
    ap.add_argument(
        '--max_images',
        type=int,
        default=int(os.environ.get('EXPL_MAX_IMAGES', '-1')),
        help='Images to explain: -1 = all, 0 = skip explanation entirely, N = '
        'first N; default from EXPL_MAX_IMAGES',
    )
    ap.add_argument('--label', action='append')
    ap.add_argument('--top_n', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    ap.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable deterministic kernels (may slow down)',
    )
    ap.add_argument(
        '--verbose', action='store_true', help='Verbose debug logging'
    )
    # Prompt configuration
    ap.add_argument(
        '--prompt_mode',
        default='unsupervised',
        choices=['unsupervised', 'binary', 'mcq'],
        help='Prompt mode to use',
    )
    ap.add_argument(
        '--prompt_label', default=None, help='Label to check for in binary mode'
    )
    ap.add_argument(
        '--choice',
        action='append',
        dest='choice_list',
        help='Add a choice label (can be repeated)',
    )
    ap.add_argument(
        '--choices',
        default=None,
        dest='choices_csv',
        help='Comma-separated choices (alternative to repeated --choice)',
    )
    ap.add_argument(
        '--choices_file',
        default=None,
        help='Choices, one per line (e.g. a class vocabulary)',
    )
    ap.add_argument(
        '--mcq_distractors',
        type=int,
        default=0,
        help='mcq: ask each image about its true class(es) plus this many '
        'random other '
        'choices (shuffled, seeded per image); 0 = always the full choice list',
    )
    # New: JSON output and data root controls
    ap.add_argument(
        '--out_json',
        default='outputs/vlm_explanations.json',
        help='Path to save JSON results',
    )
    ap.add_argument(
        '--data_root',
        default=None,
        help='Root path of dataset. If omitted, inferred from image paths',
    )
    ap.add_argument(
        '--save_only_generated_tokens',
        default=False,
        action='store_true',
        help='Save only the generated tokens in the output',
    )
    ap.add_argument(
        '--store_grounding_masks',
        default=False,
        action='store_true',
        help="Also store the concepts' segmentation masks in the JSON concept "
        'table '
        '(default: box coordinates only; masks stay in the concept bank).',
    )
    ap.add_argument(
        '--single_object',
        default=False,
        action='store_true',
        help='Set when --image_root points at single-object crops rather than '
        'multi-cell grids -- adjusts mcq/unsupervised prompt wording so it '
        "doesn't ask about non-existent grid cells.",
    )
    ap.add_argument(
        '--input_mode',
        default='image',
        choices=list(INPUT_MODES),
        help="'image': image + prompt. 'text_only': no image, the model writes "
        "each image's class name (parent folder or --label); needs "
        '--single_object.',
    )
    ap.add_argument(
        '--center_calibration',
        default=None,
        help="JSON of training images ({'images': [...], 'labels': [...]}) or "
        'a grid spec: match '
        'concepts with centred cosine, the answer-state mean taken from '
        'these images',
    )
    ap.add_argument(
        '--token_state',
        default='predicting',
        choices=['predicting', 'at'],
        help='state matched to the concepts per answer token: the one '
        'predicting it '
        '(default) or the one AT it (concepts from forced answers, '
        'FORCED_RESPONSE); '
        'the stored activations are the predicting states either way.',
    )
    args = ap.parse_args()
    if args.input_mode == 'text_only' and not args.single_object:
        ap.error(
            '--input_mode text_only needs --single_object (one class per image)'
        )

    # Configure logging for terminal output
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format='%(levelname)s: %(message)s',
    )

    # Build image list from --image_root (a directory or a JSON image/grid list)
    # if provided
    if getattr(args, 'image_root', None):
        collected = image_refs.list_refs(args.image_root)
        if args.image is None:
            args.image = []
        args.image.extend(collected)
        logging.info(
            'Discovered %s images in %s', len(collected), args.image_root
        )

    if not args.image:
        raise ValueError(
            'No images provided. Use --image one or more times, or '
            'provide --image_root.'
        )

    # --max_images / EXPL_MAX_IMAGES: -1 = all, 0 = skip explanation, N = first
    # N
    if args.max_images == 0:
        logging.info(
            'EXPL_MAX_IMAGES=0 — explanation disabled, exiting without '
            'processing.'
        )
        return
    if args.max_images > 0 and len(args.image) > args.max_images:
        logging.info(
            'Capping images to explain: %s of %s '
            '(--max_images / EXPL_MAX_IMAGES)',
            args.max_images,
            len(args.image),
        )
        args.image = args.image[: args.max_images]
        if args.label:
            args.label = args.label[: args.max_images]

    # Set seeds early for reproducibility
    set_seed(
        args.seed,
        deterministic=args.deterministic,
        disable_tf32=args.deterministic,
    )
    # add a  exception is batch size bigger than 1 the code does not work, the
    # error due the dataloader operation
    # if args.batch_size > 1:
    #    raise ValueError("Batch size greater than 1 is not supported.")
    # Prepare choices list if provided
    prompt_choices = None
    if getattr(args, 'choice_list', None):
        prompt_choices = args.choice_list
    elif getattr(args, 'choices_csv', None):
        prompt_choices = args.choices_csv
    elif args.choices_file:
        prompt_choices = [
            line.strip()
            for line in open(args.choices_file, encoding='utf-8')
            if line.strip()
        ]

    explainer = VLMConceptExplainer(
        args.model_name,
        args.concept_path,
        args.layer_path,
        verbose=args.verbose,
        prompt_mode=args.prompt_mode,
        prompt_label=args.prompt_label,
        prompt_choices=prompt_choices,
        mcq_distractors=args.mcq_distractors,
        mcq_seed=args.seed,
        save_only_generated_tokens=args.save_only_generated_tokens,
        single_object=args.single_object,
        input_mode=args.input_mode,
        token_state=args.token_state,
    )
    if args.center_calibration:
        # Centred matching (VLMConceptExplainer.set_centering): the answer-state
        # mean comes from training
        # images under the same prompt, never from the explained images.
        cal = json.load(open(args.center_calibration, encoding='utf-8'))
        if 'grids' in cal:
            cal_images, cal_labels = (
                image_refs.list_refs(args.center_calibration),
                None,
            )
        else:
            cal_images, cal_labels = cal['images'], cal.get('labels')
        logging.info(
            '[explainer] centring: calibration answers on %s training images',
            len(cal_images),
        )
        states = explainer.calibration_states(
            cal_images, cal_labels, batch_size=args.batch_size
        )
        center = states.mean(dim=0)
        explainer.set_centering(center)
        np.save(
            os.path.splitext(args.out_json)[0] + '_center.npy', center.numpy()
        )
    res = explainer.explain_with_concept(
        args.image,
        ground_truth_labels=args.label,
        top_n=args.top_n,
        batch_size=args.batch_size,
    )
    for r in res:
        logging.info(
            'Image %s -> gt=%s\nModel: %s',
            r.get('image_path'),
            r['ground_truth'],
            r['model_output'],
        )
        top_list = (
            r.get('top_concepts_over_sequence') or r.get('top_concepts') or []
        )
        logging.info('Top concepts (aggregate):')
        for c in top_list:
            logging.info(
                '  #%s idx=%s sim=%.4f text=%s',
                c['rank'],
                c['concept_index'],
                c['similarity'],
                c['text_grounding'],
            )

        # Optional: brief per-token top-1 preview + top-N concept names
        pt = r.get('per_token_concepts') or []
        if pt:
            logging.info('Per-token top-1 (preview) + top-N names:')
            for tok in pt:
                if tok.get('top_concepts'):
                    c = tok['top_concepts'][0]
                    # Build top-N concept names list
                    n_show = (
                        min(args.top_n, len(tok['top_concepts']))
                        if hasattr(args, 'top_n')
                        else len(tok['top_concepts'])
                    )
                    names = []
                    for cc in tok['top_concepts'][:n_show]:
                        nm = (
                            cc.get('concept_name')
                            or cc.get('text_grounding')
                            or f"idx{cc.get('concept_index')}"
                        )
                        names.append(str(nm))
                    names_str = ', '.join(names)
                    logging.info(
                        "  t=%s '%s' -> idx=%s text=%s sim=%.4f | top%s: %s",
                        tok['token_index'],
                        tok.get('token_text', ''),
                        c['concept_index'],
                        c.get('text_grounding'),
                        c['similarity'],
                        n_show,
                        names_str,
                    )
        logging.info('-' * 60)

    # Save consolidated JSON with metadata
    img_abs_list = [os.path.abspath(str(p)) for p in (args.image or [])]
    if args.data_root is not None:
        data_root = os.path.abspath(args.data_root)
    else:
        data_root = os.path.commonpath(img_abs_list) if img_abs_list else None

    # Helper to convert numpy/torch types for JSON serialization
    def _make_json_serializable(obj):
        """Recursively convert numpy/torch values to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: _make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            # NumPy does not support torch.bfloat16; cast tensors to float32
            # first.
            return obj.detach().to(dtype=torch.float32).cpu().numpy().tolist()
        elif hasattr(obj, 'tolist'):  # Generic fallback for array-like objects
            return obj.tolist()
        else:
            return obj

    # Clean results: pull out internal numpy embeddings into a separate stacked
    # array (real per-token activations from actual grid inference) instead of
    # discarding them -- tgcl/evaluation/concept_insertion_deletion.py edits
    # the REAL activation the model produced along the concepts, rather than
    # operating on the standalone concept vector in isolation.
    # Each token keeps an 'activation_index' pointing into that array.
    all_activations: List[np.ndarray] = []
    cleaned_results = []
    for r in res:
        r_clean = r.copy()
        if 'per_token_concepts' in r_clean:
            cleaned_tokens = []
            for tok in r_clean['per_token_concepts']:
                emb = tok.get('_token_embedding_np')
                tok_clean = {
                    k: v for k, v in tok.items() if not k.startswith('_')
                }
                if emb is not None:
                    tok_clean['activation_index'] = len(all_activations)
                    all_activations.append(np.asarray(emb, dtype=np.float32))
                cleaned_tokens.append(tok_clean)
            r_clean['per_token_concepts'] = cleaned_tokens
        cleaned_results.append(r_clean)

    activations_path = None
    if all_activations:
        activations_path = (
            os.path.splitext(args.out_json)[0] + '_activations.npy'
        )
        os.makedirs(os.path.dirname(activations_path), exist_ok=True)
        np.save(activations_path, np.stack(all_activations, axis=0))
        logging.info(
            'Saved %s real per-token activations to %s',
            len(all_activations),
            activations_path,
        )

    out_payload: Dict[str, Any] = {
        'model_card': args.model_name,
        'layer_path': args.layer_path,
        'data_root': data_root,
        'activations_path': activations_path,
        # Raw per-token hidden states at layer_path (concept_insertion_deletion
        # rejects files without this marker: older ones were token-normalized).
        'activations_kind': 'hidden_state' if activations_path else None,
        'results': _make_json_serializable(cleaned_results),
    }
    # Each concept's grounding (names, paths, box coordinates) once per file in
    # 'concept_table' instead of in every token entry; masks stay in the concept
    # bank. Readers restore the old layout with helpers.explanations.expand().
    compact(out_payload, keep_masks=args.store_grounding_masks)
    # A failed save must fail the step (no silent run without explanations).
    out_path = args.out_json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_payload, f, ensure_ascii=False, separators=(',', ':'))
    logging.info('Saved JSON to %s', out_path)

    explainer.close()

    # Proactively release GPU allocations after run
    try:
        del res
    except NameError:
        pass
    del explainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()


if __name__ == '__main__':
    main()
