"""Step 4: answer every segment with the VLM and save its hidden states.

    python -m tgcl.save_features --model_name_or_path ... --save_dir ...

The segments come from the crops JSON (step 2); each is shown masked,
inpainted and resized as configured, and the states of the answer at the
chosen layers are saved under --save_dir (one .pth per tag, named from
--save_filename).
"""

import argparse
import contextlib
import copy
import inspect
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
import torch

from tgcl.data import get_dataset_loader
from tgcl.device_utils import get_device_config
from tgcl.helpers.arguments import get_arguments
from tgcl.helpers.batching import collate_inputs as _collate_inputs
from tgcl.helpers.grounding import bbox_xywh_to_xyxy
from tgcl.helpers.layer_selection import select_layer_for_tag
from tgcl.helpers.logger import log_args
from tgcl.helpers.logger import log_num_transformer_layers
from tgcl.helpers.logger import setup_logger
from tgcl.helpers.segment_view import masked_region_view
from tgcl.helpers.segment_view import resize_to_patch
from tgcl.helpers.segment_view import segment_source
from tgcl.helpers.utils import clear_forward_hooks
from tgcl.helpers.utils import clear_hooks_variables
from tgcl.helpers.utils import compute_time_left
from tgcl.helpers.utils import feature_file_paths
from tgcl.helpers.utils import hidden_state_hooks_paused
from tgcl.helpers.utils import set_seed
from tgcl.helpers.utils import setup_hooks
from tgcl.helpers.utils import update_dict_of_list
from tgcl.helpers.vision_token_masking import build_full_keep
from tgcl.helpers.vision_token_masking import compute_token_keep_mask
from tgcl.helpers.vision_token_masking import find_vision_module
from tgcl.helpers.vision_token_masking import VisionTokenMasker
from tgcl.mask_utils import decode_mask_rle
from tgcl.models import get_model_class
from tgcl.models.constants import CONCEPT_PROMPT_TEMPLATES
from tgcl.models.image_text_model import ImageTextModel

# FORCED_RESPONSE: the class word is searched in the last FORCED_SEARCH_TAIL
# tokens of each row (the
# forced answer and the chat template's end-of-turn tokens; rows are
# left-padded).
FORCED_SEARCH_TAIL = 32


def forced_answer_fields(
    model: Any,
    model_class: Any,
    images: List[Any],
    instruction: str,
    response: str,
    pad_id: int,
) -> Dict[str, Any]:
    """One forward pass of `instruction` + the fixed assistant `response` (e.g.
    "There is a car.") per
    image, with the hidden-state hooks recording; returns the item fields the
    token-of-interest hook
    reads, so it takes the states AT the class word's tokens (mean over a
    multi-token word):
    model_output = the input ids (in a forward pass position t holds the state
    of token t) and
    model_generated_output = their last FORCED_SEARCH_TAIL ids (where the word
    is searched).

    The same text for every segment: the state differs only by the image
    (a development probe:
    real vs absent objects separable at AUC 0.90-0.91 for Gemma-3n, Qwen2-VL and
    Qwen2.5-VL).
    """
    per = [
        model_class.preprocessor(
            instruction=instruction,
            image_file=img,
            response=response,
            generation_mode=False,
        )
        for img in images
    ]
    inputs = _collate_inputs(per, pad_id=pad_id)
    keep = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )
    with torch.no_grad():
        model(**inputs, **keep)
    ids = inputs["input_ids"]
    return {
        "model_output": [ids[b] for b in range(ids.shape[0])],
        "model_generated_output": [
            ids[b, -FORCED_SEARCH_TAIL:].tolist() for b in range(ids.shape[0])
        ],
    }


# SEGMENT_RESOLUTION=original: long side of the image a segment is cut from
# (ImageNet has 4000 px photos).
def _load_selected_layers_map(
    save_dir: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Load per-tag layer selections written by the pipeline step
    ``tgcl.select_layers`` (<save_dir>/logitlens/selected_layers.json)."""
    path = os.path.join(save_dir, "logitlens", "selected_layers.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if logger is not None:
        logger.info(
            f"Loaded precomputed layer selections for {len(mapping)} "
            f"tags from {path}"
        )
    return mapping if isinstance(mapping, dict) else {}


@torch.no_grad()
def inference(
    loader: Any,
    model_class: ImageTextModel,
    hook_return_functions: Optional[List[Callable]],
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, List[Any]]:
    """Run the VLM over every segment and collect what the hooks return.

    Each batch is answered (generation, or teacher-forced on the forced
    answer); the hooks capture the hidden states of the chosen layers, and
    each hook_return_function turns them into per-sample fields.

    Args:
        loader: The segments' DataLoader.
        model_class: The VLM wrapper.
        hook_return_functions: Functions turning captured states into fields.
        logger: Progress logger.
        args: Command-line arguments.

    Returns:
        {field: [one value per sample]} (answers, hidden states, metadata).
    """
    num_iterations = len(loader)
    hook_data = {}
    model = model_class.get_model()
    # No cache for plain forward passes (teacher forcing); generate() below
    # turns it on per call unless --no_kv_cache.
    if hasattr(model, "config") and getattr(model.config, "use_cache", True):
        model.config.use_cache = False
        if logger:
            logger.info("Disabled model.config.use_cache")
    log_num_transformer_layers(model, model_name=args.model_name_or_path)

    # ------------------------------------------------------------------
    # VISION_TOKEN_MASKING=1: keep the FULL image as VLM input and zero out
    # the vision-encoder output tokens whose patch lies outside the crop
    # bbox / segmentation mask, instead of cropping/blurring pixels.
    # ------------------------------------------------------------------
    vision_token_masking = int(os.environ.get("VISION_TOKEN_MASKING", "0")) == 1
    vision_masker = None
    vision_merge_size = 2
    vision_keep_threshold = float(
        os.environ.get("VISION_TOKEN_MASK_THRESHOLD", "0.5")
    )
    if vision_token_masking:
        vision_module = find_vision_module(
            model, os.environ.get("VISION_TOKEN_MASK_MODULE") or None
        )
        if vision_module is None:
            vision_token_masking = False
            if logger:
                logger.warning(
                    "VISION_TOKEN_MASKING=1 but no vision tower found on "
                    f"{type(model).__name__} — falling back to the crop "
                    "pipeline"
                )
        else:
            vision_merge_size = int(
                getattr(vision_module, "spatial_merge_size", 2)
            )
            vision_masker = VisionTokenMasker(vision_module, logger=logger)
            if logger:
                logger.info(
                    "[vision-token-mask] Enabled: full images are passed "
                    "to the "
                    f"VLM and out-of-region vision tokens are zeroed on "
                    f"{type(vision_module).__name__} "
                    f"(merge_size={vision_merge_size}, "
                    f"threshold={vision_keep_threshold})"
                )

    # FORCED_RESPONSE (e.g. "There is a [concept]."): the generated answer
    # decides positive / negative,
    # the features come from a forced pass of FORCED_INSTRUCTION + this answer
    # (forced_answer_fields).
    forced_response = os.environ.get("FORCED_RESPONSE", "").strip()
    forced_instruction = os.environ.get(
        "FORCED_INSTRUCTION", "Describe the image."
    )
    if forced_response and vision_token_masking:
        raise ValueError(
            "FORCED_RESPONSE does not support VISION_TOKEN_MASKING=1"
        )
    if forced_response and logger:
        logger.info(
            f"[forced] features from {forced_instruction!r} + "
            f"{forced_response!r}; answers decide only"
        )

    start_time = time.time()
    for i, item in enumerate(loader):
        # Batchify: gather all texts and images in the batch
        texts = (
            item["text"] if isinstance(item["text"], list) else [item["text"]]
        )
        # Replace placeholder [concept] with --token_of_interest value if
        # provided
        toi = getattr(args, "token_of_interest", None)
        toi_str = str(toi).strip() if toi is not None else ""
        if (
            toi is not None
            and getattr(args, "prompt_template", None)
            in CONCEPT_PROMPT_TEMPLATES
        ):
            texts = [
                t.replace("[concept]", toi_str) if isinstance(t, str) else t
                for t in texts
            ]
        item["text"] = texts
        image_paths = (
            item["image"]
            if isinstance(item["image"], list)
            else [item["image"]]
        )
        img_ids = item.get("img_id", None)
        crop_locations = item.get("bbox", None)
        image_sizes = item.get("image_size", None)
        seg_mask_rles = item.get("seg_mask_rle", None)
        is_concept_flags = item.get("is_concept", None)
        patch_sizes = item.get("patch_size", None)

        # Load blur radius from env (default from .env MASK_BLUR_RADIUS)
        blur_radius_value = int(os.environ.get("MASK_BLUR_RADIUS", "10"))

        # Shared masking method for FG and BG views.
        inpainting_method = os.environ.get("INPAINTING_METHOD", "gaussian_blur")

        # Build per-sample inputs. If bboxes provided, crop and pass PIL Image;
        # else pass paths.
        per_sample_inputs: List[Dict[str, Any]] = []
        # Per-sample vision-token keep masks (VISION_TOKEN_MASKING=1); None
        # means "keep all tokens" for that sample.
        per_sample_token_keeps: List[Optional[torch.Tensor]] = []
        # The image each sample shows the VLM (a PIL segment or a path): the
        # forced pass reuses it.
        per_sample_images: List[Any] = []

        def _compute_sample_keep(sample_inputs, token_mask_np):
            """The vision-token keep mask of this sample from a full fg mask."""
            if not vision_token_masking or token_mask_np is None:
                return None
            grid = sample_inputs.get("image_grid_thw", None)
            if grid is None:
                if logger:
                    logger.warning(
                        "[vision-token-mask] processor output has no "
                        "image_grid_thw "
                        "— cannot map mask to vision tokens, keeping all tokens"
                    )
                return None
            # pylint: disable-next=possibly-used-before-assignment  (imported when masking is on)
            return compute_token_keep_mask(
                token_mask_np,
                grid[0],
                merge_size=vision_merge_size,
                threshold=vision_keep_threshold,
            )

        def _ensure_per_image_bbox_list(crops, n_imgs):
            # Accept: None, single 4-elt tuple/list, list of 4-elt per image
            if crops is None:
                return [None] * n_imgs
            if (
                isinstance(crops, (list, tuple))
                and len(crops) == 4
                and all(isinstance(v, (int, float)) for v in crops)
            ):
                return [list(crops)] * n_imgs
            # if it's a list of per-image bboxes
            if isinstance(crops, (list, tuple)) and len(crops) == n_imgs:
                return list(crops)
            # Fallback: broadcast None
            return [None] * n_imgs

        def _ensure_per_image_size_list(sizes, n_imgs):
            # Accept: None, single [w,h], list of [w,h] per image
            if sizes is None:
                return [None] * n_imgs
            if (
                isinstance(sizes, (list, tuple))
                and len(sizes) == 2
                and all(isinstance(v, (int, float)) for v in sizes)
            ):
                return [list(sizes)] * n_imgs
            if isinstance(sizes, (list, tuple)) and len(sizes) == n_imgs:
                return list(sizes)
            return [None] * n_imgs

        def _ensure_per_image_str_list(values, n_imgs):
            if values is None:
                return [None] * n_imgs
            if isinstance(values, str):
                return [values] * n_imgs
            if isinstance(values, (list, tuple)) and len(values) == n_imgs:
                return [None if v is None else str(v) for v in values]
            return [None] * n_imgs

        per_image_img_ids = _ensure_per_image_str_list(
            img_ids, len(image_paths)
        )
        per_image_bboxes = _ensure_per_image_bbox_list(
            crop_locations, len(image_paths)
        )
        per_image_sizes = _ensure_per_image_size_list(
            image_sizes, len(image_paths)
        )

        # Per-image seg mask RLE (can be None, a single dict, or a list parallel
        # to images)
        def _ensure_per_image_mask_rle(masks, n_imgs):
            if masks is None:
                return [None] * n_imgs
            if isinstance(masks, dict):
                return [masks] * n_imgs
            if isinstance(masks, (list, tuple)) and len(masks) == n_imgs:
                return list(masks)
            return [None] * n_imgs

        per_image_mask_rles = _ensure_per_image_mask_rle(
            seg_mask_rles, len(image_paths)
        )

        # Per-image is_concept flag (bool)
        def _ensure_per_image_is_concept(flags, n_imgs):
            if flags is None:
                return [False] * n_imgs
            if isinstance(flags, bool):
                return [flags] * n_imgs
            if isinstance(flags, (list, tuple)) and len(flags) == n_imgs:
                return [bool(f) for f in flags]
            return [False] * n_imgs

        per_image_is_concept = _ensure_per_image_is_concept(
            is_concept_flags, len(image_paths)
        )

        # Per-image patch_size (scalar int or list of ints parallel to images)
        def _ensure_per_image_patch_size(ps, n_imgs):
            if ps is None:
                env_ps = int(os.environ.get("PATCH_SIZE", "0"))
                return [env_ps if env_ps > 0 else None] * n_imgs
            if isinstance(ps, (int, float)):
                return [int(ps)] * n_imgs
            if isinstance(ps, (list, tuple)) and len(ps) == n_imgs:
                return [int(v) if v is not None else None for v in ps]
            if isinstance(ps, (list, tuple)) and len(ps) == 1:
                return [int(ps[0])] * n_imgs
            return [None] * n_imgs

        per_image_patch_sizes = _ensure_per_image_patch_size(
            patch_sizes, len(image_paths)
        )

        # Shared context/boundary pixels for both FG and BG views.
        context_pixels_value = int(os.environ.get("MASK_CONTEXT_PIXELS", "0"))
        boundary_pixels_value = context_pixels_value
        # MASK_FULL_IMAGE=1: whole image as input, only the outside of the
        # mask is filled (INPAINTING_METHOD, e.g. blackout); no crop/resize.
        mask_full_image = os.environ.get("MASK_FULL_IMAGE", "0") == "1"

        for idx in range(len(image_paths)):
            img_ref = image_paths[idx]
            sample_id = per_image_img_ids[idx] or f"batch{i}_idx{idx}"
            sample_id = os.path.basename(str(sample_id)).replace(os.sep, "_")
            bbox = per_image_bboxes[idx]
            mask_rle = per_image_mask_rles[idx]
            is_concept_value = per_image_is_concept[idx]
            ps_value = per_image_patch_sizes[idx]
            # Full-image fg mask used for vision-token masking (None = keep all)
            token_mask_np = None

            # ================================================================
            # MASK-CENTRIC PATH (preferred): use segmentation mask directly
            #
            # Pipeline:
            #   1. Decode RLE mask
            #   2. Find tight bounding box of the mask
            #   3. Expand bbox by MASK_CONTEXT_PIXELS (fg) or
            #      MASK_CONTEXT_PIXELS_BG (bg) for spatial context
            #   4. Crop image + mask to expanded bbox
            #   5. Inpaint outside mask WITHIN the crop
            #   6. Resize crop to patch_size × patch_size
            #   7. Pass resized crop to VLM
            # ================================================================
            if mask_rle is not None and isinstance(mask_rle, dict):
                # Select masking parameters based on fg / bg role
                eff_blur = blur_radius_value
                eff_method = inpainting_method

                if logger and idx == 0:
                    logger.info(
                        f"[mask-centric] batch {i}, idx {idx}: "
                        f"is_concept={is_concept_value}, mask_rle "
                        f"keys={list(mask_rle.keys())}, blur={eff_blur}, "
                        f"method={eff_method}, ctx={context_pixels_value}, "
                        f"boundary={boundary_pixels_value}, patch={ps_value}, "
                        f"full_image={mask_full_image}"
                    )
                img, scale = segment_source(img_ref, per_image_sizes[idx])
                # context / boundary pixels are given in the crops.json frame
                ctx_px = int(round(context_pixels_value * scale))
                bnd_px = int(round(boundary_pixels_value * scale))

                W, H = img.size

                mask_np = decode_mask_rle(mask_rle)  # bool (H, W)
                # Mask polarity is already normalised in
                # crops_to_json.py (_normalize_mask_polarity /
                # positive_negative_segment).  No alteration here —
                # save_features uses masks exactly as produced.

                # Resize mask to image size if needed
                if mask_np.shape != (H, W):
                    mask_pil = Image.fromarray(
                        mask_np.astype("uint8") * 255, mode="L"
                    )
                    mask_pil = mask_pil.resize((W, H), resample=Image.NEAREST)
                    mask_np = np.array(mask_pil) > 127

                # Step 2: Find tight bounding box of the mask
                ys, _ = np.where(mask_np)
                if len(ys) == 0:
                    # Empty mask — fall back to full image
                    raise ValueError("Mask is empty (0 foreground pixels)")

                # ------------------------------------------------------
                # VISION-TOKEN MASKING PATH: keep the full image as VLM
                # input; the region selection happens after the vision
                # encoder by zeroing out-of-mask tokens. Apply the same
                # boundary semantics as the crop path (fg → dilate,
                # bg → erode) directly in full-image space.
                # ------------------------------------------------------
                if vision_token_masking:
                    if bnd_px > 0:
                        struct = np.ones(
                            (2 * bnd_px + 1, 2 * bnd_px + 1),
                            dtype=bool,
                        )
                        if is_concept_value:
                            mask_np = binary_dilation(
                                mask_np,
                                structure=struct,
                                iterations=1,
                            ).astype(bool)
                        else:
                            eroded = binary_erosion(
                                mask_np,
                                structure=struct,
                                iterations=1,
                            )
                            if eroded is not None and eroded.any():
                                mask_np = eroded.astype(bool)
                    token_mask_np = mask_np
                    vlm_img = img

                else:
                    # Steps 3-6: crop to the mask box (or keep the full
                    # image with MASK_FULL_IMAGE=1), fill outside the
                    # mask, resize to the patch size.
                    vlm_img, _ = masked_region_view(
                        img,
                        mask_np,
                        is_concept=is_concept_value,
                        method=eff_method,
                        blur_radius=eff_blur,
                        context_pixels=ctx_px,
                        boundary_pixels=bnd_px,
                        patch_size=ps_value,
                        full_image=mask_full_image,
                    )

                sample_inputs_value = model_class.preprocessor(
                    instruction=texts[idx],
                    image_file=vlm_img,
                    response="",
                    generation_mode=args.generation_mode,
                )
                per_sample_inputs.append(sample_inputs_value)
                per_sample_images.append(vlm_img)
                per_sample_token_keeps.append(
                    _compute_sample_keep(sample_inputs_value, token_mask_np)
                )

            # ================================================================
            # LEGACY BBOX PATH (fallback when no mask available)
            # ================================================================
            elif bbox is not None:
                img, scale = segment_source(img_ref, per_image_sizes[idx])

                W, H = img.size
                x1, y1, x2, y2 = bbox_xywh_to_xyxy(
                    [v * scale for v in bbox], W, H
                )

                if vision_token_masking:
                    # Full image in, bbox becomes a rectangular token mask.
                    token_mask_np = np.zeros((H, W), dtype=bool)
                    token_mask_np[y1:y2, x1:x2] = True
                    crop_img = img
                else:
                    crop_img = img.crop((x1, y1, x2, y2))

                if not vision_token_masking:
                    crop_img = resize_to_patch(crop_img, ps_value)

                sample_inputs_value = model_class.preprocessor(
                    instruction=texts[idx],
                    image_file=crop_img,
                    response="",
                    generation_mode=args.generation_mode,
                )
                per_sample_inputs.append(sample_inputs_value)
                per_sample_images.append(crop_img)
                per_sample_token_keeps.append(
                    _compute_sample_keep(sample_inputs_value, token_mask_np)
                )
            else:
                # no bbox -> default behavior
                per_sample_inputs.append(
                    model_class.preprocessor(
                        instruction=texts[idx],
                        image_file=image_paths[idx],
                        response="",
                        generation_mode=args.generation_mode,
                    )
                )
                per_sample_images.append(image_paths[idx])
                per_sample_token_keeps.append(None)

        pad_id = getattr(model_class.get_tokenizer(), "pad_token_id", 0) or 0
        inputs = _collate_inputs(per_sample_inputs, pad_id=pad_id)

        if vision_masker is not None:
            keep_full = build_full_keep(  # pylint: disable=possibly-used-before-assignment
                per_sample_token_keeps,
                inputs.get("image_grid_thw", None),
                merge_size=vision_merge_size,
                logger=logger,
            )
            vision_masker.set_keep(keep_full)
            if logger and i == 0 and keep_full is not None:
                logger.info(
                    f"[vision-token-mask] batch 0: keeping "
                    f"{int(keep_full.sum())}/{keep_full.numel()} vision "
                    "tokens"
                )

        try:
            if args.generation_mode:
                # With the KV cache the hidden-state hook sees the prompt once
                # and
                # then one new position per step; get_hidden_states concatenates
                # them into the same (B, L + T - 1, D) sequence a cache-free
                # forward gives. Without it every step re-runs (and the hook
                # copies to the CPU) the whole sequence: several times slower.
                use_kv_cache = not getattr(args, "no_kv_cache", False)
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=use_kv_cache,
                )
                if use_kv_cache:
                    # The hook appends to a global list, which makes a compiled
                    # forward (Gemma-3n's static cache) recompile every token.
                    gen_kwargs["disable_compile"] = True
                # FORCED_RESPONSE: the answer only decides positive / negative;
                # the
                # states come from the forced pass below, so the hooks skip
                # generation.
                with (
                    hidden_state_hooks_paused()
                    if forced_response
                    else contextlib.nullcontext()
                ):
                    out = model.generate(**inputs, **gen_kwargs)
            else:
                out = model(**inputs).logits
        finally:
            if vision_masker is not None:
                vision_masker.clear()

        # Debatch outputs: store per-sample outputs as a list for downstream
        if isinstance(out, torch.Tensor) and out.dim() >= 1 and out.size(0) > 1:
            item["model_output"] = [out[b] for b in range(out.size(0))]
        else:
            item["model_output"] = out
        # Keep using `out` locally for subsequent computations

        item["concept_name"] = toi_str
        if args.generation_mode:
            # `out` holds generated token ids only in generation mode; in
            # teacher-forcing mode it is logits and cannot be decoded.
            input_ids_tensor = inputs["input_ids"]
            if input_ids_tensor.ndim == 1:
                input_ids_tensor = input_ids_tensor.unsqueeze(0)
            B = input_ids_tensor.shape[0]
            input_lens: List[int] = []
            for b in range(B):
                row = input_ids_tensor[b]
                input_lens.append(row.shape[0])
            # Slice generated tokens per sample
            generated_ids = [
                out[b, input_lens[b] :] for b in range(out.size(0))
            ]
            # Debatch to plain Python lists of token ids for portability
            model_generated_output_list = [
                t.tolist() if torch.is_tensor(t) else list(t)
                for t in generated_ids
            ]
            item["model_generated_output"] = model_generated_output_list
            item[
                "model_predictions"
            ] = model_class.get_tokenizer().batch_decode(
                model_generated_output_list, skip_special_tokens=True
            )

        hook_item = item
        if forced_response and args.generation_mode:
            clear_hooks_variables()
            hook_item = dict(
                item,
                **forced_answer_fields(
                    model,
                    model_class,
                    per_sample_images,
                    forced_instruction,
                    forced_response.replace("[concept]", toi_str),
                    pad_id,
                ),
            )

        if hook_return_functions is not None:
            for func in hook_return_functions:
                if func is not None:
                    hook_output = func(**hook_item)
                    if hook_output:
                        item.update(hook_output)

        hook_data = update_dict_of_list(item, hook_data)
        clear_hooks_variables()
        # Iteration-end cleanup to reduce GPU memory usage
        del out  # model outputs
        del inputs  # batched inputs
        # Drop potentially large GPU-backed fields from item now that data is
        # aggregated
        if isinstance(item, dict) and "model_output" in item:
            item["model_output"] = None
        if (i + 1) % 100 == 0:
            time_left = compute_time_left(start_time, i, num_iterations)
            logger.info(
                f"Iteration: {i}/{num_iterations},  Estimated time left: "
                f"{time_left:.2f} mins"
            )
    if vision_masker is not None:
        vision_masker.remove()
    return hook_data


def main():
    """Main entry point for save_features script."""
    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, "logs.log"))

    set_seed(args.seed)

    logger.info("Loading model: %s", args.model_name_or_path)
    log_args(args, logger)

    device_config = get_device_config(args.device)
    device = device_config.primary_device
    logger.info(
        "Device config: %s -> primary=%s, gpu_ids=%s",
        device_config.raw,
        device,
        device_config.gpu_ids,
    )

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
        device_config=device_config,
    )

    hook_return_functions, hook_postprocessing_functions = None, None
    if args.dataset_name != "json_crop_map":
        # json_crop_map re-registers hooks per tag below (clearing these
        # first), so this initial registration would only be wasted work --
        # and, for save_hidden_states_for_token_of_interest, would crash here
        # since args.token_of_interest is a per-tag key set inside that loop,
        # not yet available at this point.
        hook_return_functions, hook_postprocessing_functions = setup_hooks(
            model=model_class.model_,
            modules_to_hook=args.modules_to_hook,
            hook_names=args.hook_names,
            tokenizer=model_class.get_tokenizer(),
            logger=logger,
            args=args,
        )
    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args
    )
    # if loader is a list run inference on each and aggregate the results and
    # update the args accordingly
    if args.dataset_name == "json_crop_map":
        # An empty loader dict would make the per-tag loop below a silent
        # no-op: the script would exit 0 without saving a single feature file
        # and the pipeline would only fail (confusingly) at the decompose step.
        if not loader:
            raise RuntimeError(
                f"json_crop_map produced 0 concept splits from "
                f"'{args.annotation_file}' — no features to save. Check that "
                "the crops JSON has concept entries (pipeline steps 1-2) and "
                "that concept filtering did not drop every tag."
            )
        base_save_filename = getattr(args, "save_filename", None)
        base_modules_to_hook = copy.deepcopy(args.modules_to_hook)
        # Per-tag layer selections computed by the pipeline step
        # `src/select_layers.py` (runs after crops.json is built).
        selected_layers_map = _load_selected_layers_map(
            args.save_dir, logger=logger
        )

        for key, ld in loader.items():

            # Set before setup_hooks() below: the
            # save_hidden_states_for_token_of_interest
            # hook reads args.token_of_interest at registration time, so it must
            # already be this tag's key when hooks are set up, not the previous
            # tag's (or unset on the first iteration).
            args.token_of_interest = key
            if base_save_filename:
                args.save_filename = f"{base_save_filename}_{key}"
            else:
                args.save_filename = f"token_of_interest_concept_split_{key}"

            # Resume: tags saved by an interrupted run are kept (files are
            # written atomically, so an existing one is complete).
            done_files = feature_file_paths(args)
            if done_files and all(os.path.exists(p) for p in done_files):
                logger.info("Skip tag '%s' (found %s)", key, done_files[0])
                continue

            # Per-tag layer selection (LOGIT_LENS_LAYER_SELECTION=1): hook the
            # layer chosen by the logit-lens sweep instead of the static
            # LAYER_PATH. Prefer the precomputed selected_layers.json from the
            # pipeline step; when running save_features standalone without
            # that file, fall back to sweeping inline. Falls back to the
            # configured modules when disabled or when selection fails.
            args.modules_to_hook = copy.deepcopy(base_modules_to_hook)
            precomputed = selected_layers_map.get(str(key), None)
            if isinstance(precomputed, dict) and precomputed.get(
                "selected_layer"
            ):
                layer_selection = {
                    "selected_layer": precomputed["selected_layer"],
                    "summary": precomputed,
                }
                logger.info(
                    "[layer-selection] Using precomputed layer for '%s': %s",
                    key,
                    precomputed["selected_layer"],
                )
            else:
                layer_selection = select_layer_for_tag(
                    model_class=model_class,
                    tag=key,
                    dataset=ld.dataset,
                    args=args,
                    logger=logger,
                )
            if layer_selection is not None:
                args.modules_to_hook = [[layer_selection["selected_layer"]]]
                args.exact_match_modules_to_hook = True

            # Remove hooks from previous tags (and from the initial
            # setup_hooks above) before registering this tag's hooks, so each
            # module is hooked exactly once.
            clear_forward_hooks(model_class.model_)
            clear_hooks_variables()

            hook_return_functions, hook_postprocessing_functions = setup_hooks(
                model=model_class.model_,
                modules_to_hook=args.modules_to_hook,
                hook_names=args.hook_names,
                tokenizer=model_class.get_tokenizer(),
                logger=logger,
                args=args,
            )
            logger.info("Running inference on dataset split: %s", key)
            hook_data = inference(
                loader=ld,
                model_class=model_class,
                hook_return_functions=hook_return_functions,
                logger=logger,
                args=args,
            )
            # Record which layer produced this tag's features so downstream
            # stages (decomposition, concept bank) stay traceable per tag.
            try:
                hooked_layer = args.modules_to_hook[0][0]
            except (TypeError, IndexError):
                hooked_layer = None
            hook_data["selected_layer"] = (
                layer_selection["selected_layer"]
                if layer_selection
                else hooked_layer
            )
            if layer_selection is not None:
                hook_data["logit_lens_selection"] = layer_selection["summary"]
            clear_forward_hooks(model_class.model_)
            if hook_postprocessing_functions is not None:
                for func in hook_postprocessing_functions:
                    if func is not None:
                        func(data=hook_data, args=args, logger=logger)

        # Backstop: the whole point of this script is to write feature files;
        # exiting 0 without any would only surface as a confusing failure in
        # the decompose step.
        features_dir = os.path.join(args.save_dir, "features")
        if not (
            os.path.isdir(features_dir)
            and any(f.endswith(".pth") for f in os.listdir(features_dir))
        ):
            raise RuntimeError(
                f"save_features finished without writing any .pth file under "
                f"'{features_dir}'. Check the hook configuration "
                f"(--hook_names {args.hook_names}) and the logs above."
            )

    else:

        hook_data = inference(
            loader=loader,
            model_class=model_class,
            hook_return_functions=hook_return_functions,
            logger=logger,
            args=args,
        )

        clear_forward_hooks(model_class.model_)
        if hook_postprocessing_functions is not None:
            for func in hook_postprocessing_functions:
                if func is not None:
                    func(data=hook_data, args=args, logger=logger)


if __name__ == "__main__":
    main()
