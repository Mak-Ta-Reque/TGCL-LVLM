"""
crops_to_json.py — Simplified mask-centric JSON builder.

Given a JSON mapping ``{tag: [relative_image_paths...]}`` and an image root,
produce a crops JSON with per-image segmentation masks in RLE format.

For each (tag, image):
  1. Concept masks   — text-prompted segmentation using *tag* as the prompt.
  2. Extra masks     — prompt-free automatic segmentation (no text prompt).
  3. Enforce limits  — ``MASKS_PER_IMAGE`` total, ``CONCEPT_MASKS_PER_IMAGE``
                       concept masks.  Concept masks are always preserved.

Supported detectors: ``none``, ``langsam``.
Image resizing (``--image_size_width``) is applied before detection for memory.
When ``detector="none"``, the pipeline skips localization and emits random
bbox-only crops. Set ``RLE=0`` in the environment to skip mask serialization.

Usage::

    python preprocessing/crops_to_json.py \\
        --mapping_json  outputs/bird2/concept_image_mapping.json \\
        --image_root    data/train/bird \\
        --output_json   outputs/bird2/crops.json \\
        --detector      langsam \\
        --masks_per_image 5 \\
        --concept_masks_per_image 1 \\
        --image_size_width 512

"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import torch
from tqdm import tqdm

from tgcl.device_utils import get_device_config
from tgcl.helpers.utils import apply_determinism_setting
from tgcl.langsam_utils import load_langsam
from tgcl.langsam_utils import predict_all_masks_langsam
from tgcl.langsam_utils import predict_bboxes_and_masks_for_tag_batched
from tgcl.mask_utils import encode_mask_rle

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _compute_virtual_resize(
    orig_w: int,
    orig_h: int,
    ref_width: Optional[int],
) -> Tuple[int, int, float]:
    """Compute target ``(w, h, scale)`` keeping aspect ratio."""
    if ref_width is None or ref_width <= 0 or orig_w <= 0 or orig_h <= 0:
        return orig_w, orig_h, 1.0
    scale = ref_width / float(orig_w)
    new_h = max(1, int(round(orig_h * scale)))
    return int(ref_width), new_h, float(scale)


def _load_and_resize(path: str, ref_width: Optional[int]):
    """Load image, optionally resize.  Returns ``(pil_img, (w,h), scale)``."""
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    new_w, new_h, scale = _compute_virtual_resize(orig_w, orig_h, ref_width)
    if ref_width and (new_w, new_h) != (orig_w, orig_h):
        resample = getattr(
            getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC
        )
        img = img.resize((new_w, new_h), resample=resample)
    return img, (new_w, new_h), scale


def _cleanup_gpu():
    """Release cached GPU memory between chunks. No gc.collect() here: it walks
    every live object, and the accumulated crop records (millions for large
    datasets) made a per-chunk collection the slowest part of this step; tensors
    are freed by reference counting. A full collection runs per tag."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# RLE encoding
# ---------------------------------------------------------------------------


def _encode_mask_rle(mask):
    """RLE-encode a binary mask for JSON serialisation."""
    return encode_mask_rle(mask)


def _random_bbox_for_image(
    image_size: Tuple[int, int],
    rng: random.Random,
    fixed_size: Optional[int] = None,
) -> List[int]:
    """Sample a random crop bbox in ``[x, y, w, h]`` format.

    If ``fixed_size`` is provided and > 0, a square crop of that size is used
    (clamped to image bounds).
    """
    width, height = image_size
    if width <= 1 or height <= 1:
        return [0, 0, max(1, width), max(1, height)]

    if fixed_size is not None and fixed_size > 0:
        crop_width = max(1, min(width, int(fixed_size)))
        crop_height = max(1, min(height, int(fixed_size)))
    else:
        area = width * height
        target_area = rng.uniform(0.15, 0.55) * area
        aspect_ratio = rng.uniform(0.75, 1.33)

        crop_width = int(round((target_area * aspect_ratio) ** 0.5))
        crop_height = int(round((target_area / aspect_ratio) ** 0.5))
        crop_width = max(1, min(width, crop_width))
        crop_height = max(1, min(height, crop_height))

    max_x = max(0, width - crop_width)
    max_y = max(0, height - crop_height)
    x = rng.randint(0, max_x) if max_x > 0 else 0
    y = rng.randint(0, max_y) if max_y > 0 else 0
    return [int(x), int(y), int(crop_width), int(crop_height)]


# ---------------------------------------------------------------------------
# Model loaders (singletons)
# ---------------------------------------------------------------------------

_LANGSAM_MODEL = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_MODEL_NAME = None
_CLIP_MODEL_DEVICE = None


def _load_langsam(device: Optional[str] = None):
    global _LANGSAM_MODEL
    if _LANGSAM_MODEL is not None:
        return _LANGSAM_MODEL
    _LANGSAM_MODEL = load_langsam(device=device)
    return _LANGSAM_MODEL


def _load_clip_model(
    model_name: str = "ViT-B/32", device: Optional[str] = None
):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_MODEL_NAME, _CLIP_MODEL_DEVICE
    import clip

    resolved_device = device or os.environ.get("CLIP_DEVICE")
    if resolved_device is None:
        # The pipeline-wide DEVICE setting (.env), not cuda:0.
        resolved_device = str(get_device_config(None).primary_device)

    if (
        _CLIP_MODEL is not None
        and _CLIP_PREPROCESS is not None
        and _CLIP_MODEL_NAME == model_name
        and _CLIP_MODEL_DEVICE == resolved_device
    ):
        return _CLIP_MODEL, _CLIP_PREPROCESS, resolved_device

    _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(
        model_name, device=resolved_device, jit=False
    )
    _CLIP_MODEL.eval()
    _CLIP_MODEL_NAME = model_name
    _CLIP_MODEL_DEVICE = resolved_device
    return _CLIP_MODEL, _CLIP_PREPROCESS, resolved_device


def _encode_clip_images(
    images: List[Any],
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
):
    if not images:
        return torch.empty((0, 0), dtype=torch.float32)

    embeddings = []
    for start in range(0, len(images), max(1, batch_size)):
        batch = images[start : start + max(1, batch_size)]
        batch_tensor = torch.stack([preprocess(img) for img in batch]).to(
            device
        )
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        embeddings.append(feats.detach().cpu())
    return torch.cat(embeddings, dim=0)


def _encode_clip_text(tag: str, model, device: str) -> "torch.Tensor":
    """CLIP text embedding for a tag, L2-normalized. Used to make sliding_window
    /random crop selection tag-aware (score candidates against the actual
    concept phrase) instead of purely geometric/diversity-based, which
    previously selected crops with zero regard for whether they depicted the tag
    at all -- the root cause of most tags failing purity re-confirmation for
    these crop modes."""
    import clip as _clip

    prompt = f"a photo of a {tag}"
    with torch.no_grad():
        tokens = _clip.tokenize([prompt]).to(device)
        feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return feat.detach().cpu()[0]


def _select_top_by_tag_relevance(
    candidate_images: List[Any],
    tag_embedding,
    model,
    preprocess,
    device: str,
    keep_n: int,
) -> List[int]:
    """Return indices of the top `keep_n` candidate_images by CLIP text-image
    cosine similarity to tag_embedding, descending."""
    embs = _encode_clip_images(candidate_images, model, preprocess, device)
    if embs.numel() == 0:
        return []
    scores = (embs @ tag_embedding.to(embs.dtype)).numpy()
    order = np.argsort(-scores)
    return order[:keep_n].tolist()


def _greedy_filter_candidates_by_clip_similarity(
    candidate_pairs: List[Tuple[List[int], Any]],
    candidate_embeddings,
    similarity_threshold: float = 0.5,
    existing_embeddings: Optional[List[List[float]]] = None,
):
    kept_pairs: List[Tuple[List[int], Any]] = []
    kept_embeddings: List[List[float]] = []

    def _to_numpy_1d(vec) -> np.ndarray:
        """Convert tensor / list / ndarray to a 1-D float32 numpy vector."""
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().numpy()
        elif not isinstance(vec, np.ndarray):
            vec = np.asarray(vec, dtype=np.float32)
        return vec.astype(np.float32).ravel()

    # Shuffle candidates so the greedy pass does not always favour the first
    # crop in arrival order (e.g. top-left in sliding-window, or generation
    # order in random mode).  After shuffling, greedy selection is unbiased.
    n = len(candidate_pairs)
    if n > 1:
        perm = np.random.permutation(n)
        candidate_pairs = [candidate_pairs[i] for i in perm]
        # Tensors and ndarrays support fancy indexing; plain sequences do not.
        if not hasattr(candidate_embeddings, "shape"):
            candidate_embeddings = np.asarray(
                candidate_embeddings, dtype=np.float32
            )
        candidate_embeddings = candidate_embeddings[perm]

    # Build a matrix of already-accepted embeddings: shape (N_seen, D).
    # Embeddings are L2-normalised upstream, so mat @ vec == cosine
    # similarities.
    seen_rows: List[np.ndarray] = []
    if existing_embeddings:
        for e in existing_embeddings:
            seen_rows.append(_to_numpy_1d(e))

    for (bbox, mask), emb in zip(candidate_pairs, candidate_embeddings):
        emb_vec = _to_numpy_1d(emb)

        if seen_rows:
            # Stack once per candidate: (N_seen, D) @ (D,) → (N_seen,) cosine
            # sims
            seen_mat = np.stack(seen_rows, axis=0)  # (N_seen, D)
            sims = seen_mat @ emb_vec  # (N_seen,) — fast BLAS
            if float(sims.max()) >= similarity_threshold:
                continue

        kept_pairs.append((bbox, mask))
        kept_embeddings.append(emb_vec.tolist())
        seen_rows.append(emb_vec)

    return kept_pairs, kept_embeddings


def _build_diverse_random_pairs_for_image(
    image,
    image_size: Tuple[int, int],
    count: int,
    candidate_count: Optional[int],
    fixed_crop_size: Optional[int],
    rng: random.Random,
    clip_model,
    clip_preprocess,
    clip_device: str,
    similarity_threshold: float = 0.5,
    candidate_batch_size: int = 32,
    max_attempts_multiplier: int = 20,
) -> List[Tuple[List[int], Any]]:
    """Build random bbox-only crops and drop near-duplicates with CLIP."""
    if count <= 0:
        count = 1
    if candidate_count is None or candidate_count <= 0:
        candidate_count = count
    candidate_count = max(candidate_count, count)

    accepted_pairs: List[Tuple[List[int], Any]] = []
    accepted_embeddings: List[List[float]] = []
    attempts = 0
    max_attempts = max(
        candidate_count * max_attempts_multiplier, candidate_batch_size
    )

    while len(accepted_pairs) < count and attempts < max_attempts:
        batch_size = min(candidate_batch_size, max_attempts - attempts)
        candidate_pairs: List[Tuple[List[int], Any]] = []
        candidate_images = []

        for _ in range(batch_size):
            bbox = _random_bbox_for_image(
                image_size,
                rng,
                fixed_size=fixed_crop_size,
            )
            x, y, w, h = bbox
            candidate_pairs.append((bbox, None))
            candidate_images.append(image.crop((x, y, x + w, y + h)))

        candidate_embeddings = _encode_clip_images(
            candidate_images,
            clip_model,
            clip_preprocess,
            clip_device,
            batch_size=min(candidate_batch_size, len(candidate_images)),
        )

        kept_pairs_batch, kept_embeddings_batch = (
            _greedy_filter_candidates_by_clip_similarity(
                candidate_pairs,
                candidate_embeddings,
                similarity_threshold=similarity_threshold,
                existing_embeddings=accepted_embeddings,
            )
        )

        for pair, emb in zip(kept_pairs_batch, kept_embeddings_batch):
            accepted_pairs.append(pair)
            accepted_embeddings.append(emb)
            if len(accepted_pairs) >= count:
                break

        attempts += batch_size

    return accepted_pairs


def _sliding_window_bboxes_for_image(
    image_size: Tuple[int, int],
    window_size_ratio_range: Tuple[float, float] = (0.15, 0.55),
    stride_ratio: float = 0.3,
    window_size: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """Generate sliding window crop bboxes in ``[x, y, w, h]`` format.

    Args:
        image_size: (width, height) of the image
        window_size_ratio_range: (min_ratio, max_ratio) of window size relative
            to image area. Ignored when ``window_size`` is given.
        stride_ratio: stride as fraction of window width (controls overlap)
        window_size: fixed, absolute window side length in pixels (square
            window). Takes precedence over window_size_ratio_range — e.g. set
            this to PATCH_SIZE to get small, patch-sized windows so far more
            distinct crops fit per image instead of a few huge, image-fraction
            windows.

    Returns:
        List of [x, y, w, h] bboxes covering the image systematically
    """
    width, height = image_size
    if width <= 1 or height <= 1:
        return [[0, 0, max(1, width), max(1, height)]]

    if window_size is not None:
        window_width = window_height = int(window_size)
    else:
        # Use midpoint of window size range for consistency
        target_area = (
            (window_size_ratio_range[0] + window_size_ratio_range[1]) / 2
        ) * (width * height)
        aspect_ratio = 1.0  # Use square windows for sliding
        window_width = int(round((target_area * aspect_ratio) ** 0.5))
        window_height = int(round((target_area / aspect_ratio) ** 0.5))
    window_width = max(1, min(width, window_width))
    window_height = max(1, min(height, window_height))

    stride = max(1, int(window_width * stride_ratio))

    bboxes: List[List[int]] = []
    bbox_set = set()

    def _append_unique_bbox(bbox: List[int]) -> None:
        key = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if key in bbox_set:
            return
        bbox_set.add(key)
        bboxes.append([key[0], key[1], key[2], key[3]])

    y = last_y = 0
    while y + window_height <= height:
        x = last_x = 0
        while x + window_width <= width:
            _append_unique_bbox([x, y, window_width, window_height])
            last_x = x
            x += stride
        last_y = y
        y += stride

    # Only add an extra, overlapping edge window when the strip the main
    # grid left uncovered is more than 25% of the window size — a smaller
    # leftover is already adequately covered by the overlap from stride, and
    # doesn't need a whole extra, heavily-overlapping window on top.
    edge_threshold_ratio = 0.25
    if len(bboxes) == 0:
        _append_unique_bbox([0, 0, max(1, width), max(1, height)])
    else:
        remaining_x = width - (last_x + window_width)
        remaining_y = height - (last_y + window_height)
        add_right_edge = remaining_x > edge_threshold_ratio * window_width
        add_bottom_edge = remaining_y > edge_threshold_ratio * window_height

        # Right edge
        if add_right_edge:
            x_edge = max(0, width - window_width)
            y_edge = 0
            while y_edge + window_height <= height:
                bbox = [x_edge, y_edge, window_width, window_height]
                _append_unique_bbox(bbox)
                y_edge += stride
        # Bottom edge
        if add_bottom_edge:
            y_edge = max(0, height - window_height)
            x_edge = 0
            while x_edge + window_width <= width:
                bbox = [x_edge, y_edge, window_width, window_height]
                _append_unique_bbox(bbox)
                x_edge += stride
        # Bottom-right corner: the right-edge loop only varies y and the
        # bottom-edge loop only varies x, so neither ever lands on
        # (width - window_width, height - window_height). Only needed when
        # both dimensions have a significant leftover — otherwise the edge
        # loop for the dimension that does gets close enough already.
        if add_right_edge and add_bottom_edge:
            _append_unique_bbox(
                [
                    max(0, width - window_width),
                    max(0, height - window_height),
                    window_width,
                    window_height,
                ]
            )

    return bboxes


def _build_sliding_window_pairs_for_image(
    image,
    image_size: Tuple[int, int],
    count: int,
    clip_model,
    clip_preprocess,
    clip_device: str,
    similarity_threshold: float = 0.5,
    window_size_ratio_range: Tuple[float, float] = (0.15, 0.55),
    stride_ratio: float = 0.3,
    candidate_batch_size: int = 32,
    window_size: Optional[int] = None,
    tag: Optional[str] = None,
    relevance_oversample: int = 4,
) -> List[Tuple[List[int], Any]]:
    """Build sliding window bbox-only crops and drop near-duplicates with CLIP.

    Args:
        image: PIL Image
        image_size: (width, height) of image
        count: number of crops to return
        clip_model: CLIP model instance
        clip_preprocess: CLIP preprocessing function
        clip_device: device for CLIP inference
        similarity_threshold: skip crops with CLIP cosine similarity >= this
        value
        window_size_ratio_range: (min_ratio, max_ratio) for window size.
            Ignored when window_size is given.
        stride_ratio: stride as fraction of window width
        candidate_batch_size: batch size for CLIP encoding
        window_size: fixed, absolute square window side length in pixels
            (e.g. PATCH_SIZE). Takes precedence over window_size_ratio_range.
        tag: concept phrase (e.g. category name) to score candidates against
            via CLIP text-image similarity. Without this, candidate windows
            are purely geometric with a diversity-only filter and carry no
            information about whether they actually depict the tag -- the
            root cause of sliding_window crops failing purity
            re-confirmation far more often than real detectors. When given,
            windows are first ranked by relevance to the tag, and only the
            top `count * relevance_oversample` most relevant ones go through
            the existing diversity dedup.
        relevance_oversample: how many top-relevance candidates (relative to
            `count`) to keep before diversity-deduping, when `tag` is given.

    Returns:
        List of (bbox, None) tuples where bbox is [x, y, w, h]
    """
    if count <= 0:
        count = 1

    # Generate all sliding window bboxes
    all_bboxes = _sliding_window_bboxes_for_image(
        image_size,
        window_size_ratio_range=window_size_ratio_range,
        stride_ratio=stride_ratio,
        window_size=window_size,
    )

    if len(all_bboxes) == 0:
        return [([0, 0, image_size[0], image_size[1]], None)]

    # Take up to 'count' bboxes if fewer available, otherwise process in batches
    # with CLIP filtering
    if len(all_bboxes) <= count:
        return [(bbox, None) for bbox in all_bboxes]

    if tag:
        # Tag-aware path: score every candidate against the tag phrase, keep
        # only the top-scoring subset, then diversity-dedup within that
        # subset (instead of deduping across ALL geometric candidates with
        # no regard for whether they show the tag at all).
        candidate_images_all = [
            image.crop((x, y, x + w, y + h)) for (x, y, w, h) in all_bboxes
        ]
        tag_embedding = _encode_clip_text(tag, clip_model, clip_device)
        keep_n = min(len(all_bboxes), max(count, count * relevance_oversample))
        top_indices = _select_top_by_tag_relevance(
            candidate_images_all,
            tag_embedding,
            clip_model,
            clip_preprocess,
            clip_device,
            keep_n=keep_n,
        )
        all_bboxes = [all_bboxes[i] for i in top_indices]

    # Build crops with CLIP filtering
    accepted_pairs: List[Tuple[List[int], Any]] = []
    accepted_embeddings: List[List[float]] = []

    for batch_start_idx in range(0, len(all_bboxes), candidate_batch_size):
        batch_end_idx = min(
            batch_start_idx + candidate_batch_size, len(all_bboxes)
        )
        batch_bboxes = all_bboxes[batch_start_idx:batch_end_idx]
        candidate_images = []

        for bbox in batch_bboxes:
            x, y, w, h = bbox
            candidate_images.append(image.crop((x, y, x + w, y + h)))

        # Encode candidates with CLIP
        candidate_embeddings = _encode_clip_images(
            candidate_images,
            clip_model,
            clip_preprocess,
            clip_device,
            batch_size=candidate_batch_size,
        )

        # Filter using greedy approach
        kept_pairs_batch, kept_embeddings_batch = (
            _greedy_filter_candidates_by_clip_similarity(
                [(bbox, None) for bbox in batch_bboxes],
                candidate_embeddings,
                similarity_threshold=similarity_threshold,
                existing_embeddings=accepted_embeddings,
            )
        )

        for pair, emb in zip(kept_pairs_batch, kept_embeddings_batch):
            accepted_pairs.append(pair)
            accepted_embeddings.append(emb)
            if len(accepted_pairs) >= count:
                return accepted_pairs

    return accepted_pairs


# ---------------------------------------------------------------------------
# Mask area helper
# ---------------------------------------------------------------------------


def _mask_area(mask) -> int:
    if mask is None:
        return 0
    return int(mask.sum())


def _normalize_mask_polarity(mask, max_fg_ratio: float = 0.50):
    """Invert a mask if it covers more than *max_fg_ratio* of the image.

    Automatic mask generation sometimes returns masks whose
    foreground represents the **background** of the scene (e.g. sky, ground)
    rather than a distinct object.  These masks have a very high
    foreground-to-total ratio.  Inverting them turns them into the smaller,
    more meaningful complementary segment.

    The bbox is recomputed after inversion.

    Args:
        mask: Binary numpy array (H, W).
        max_fg_ratio: If the foreground ratio exceeds this value the mask
            is inverted.  Default 0.50.

    Returns:
        The (possibly inverted) mask.
    """
    if mask is None:
        return mask
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return mask
    fg = int(mask.sum())
    if fg / total > max_fg_ratio:
        return ~mask
    return mask


# ---------------------------------------------------------------------------
# Subtract overlap helper
# ---------------------------------------------------------------------------


def _resize_mask_like(mask, ref):
    """mask at ref's (H, W) (nearest neighbour, boolean); unchanged if equal."""
    if mask.shape == ref.shape:
        return mask
    h, w = ref.shape
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (w, h), Image.NEAREST
    )
    return np.array(resized) > 127


def _subtract_mask(base, sub):
    """Remove *sub* pixels from *base*.  Handles resolution mismatch."""
    if base is None or sub is None:
        return base
    return base & (~_resize_mask_like(sub, base))


def _is_mask_solid(mask, max_edge_ratio: float = 0.70) -> bool:
    """Return True if *mask* is a solid filled region, not just edges/outlines.

    A mask is considered edge-like when most of its pixels disappear after
    a mild erosion — i.e. it is made of thin 1–2 px strands rather than a
    filled blob.  Such masks arise when large overlapping masks are
    subtracted from each other, leaving only the boundary residuals.

    Args:
        mask: Binary numpy array (H, W).
        max_edge_ratio: Maximum fraction of pixels that may be "edge-only"
            (removed by erosion).  Masks above this are rejected.
    """
    if mask is None:
        return False
    filled = int(mask.sum())
    if filled == 0:
        return False
    interior = int(binary_erosion(mask, iterations=2).sum())
    edge_ratio = 1.0 - (interior / filled)
    return edge_ratio <= max_edge_ratio


# ---------------------------------------------------------------------------
# Concept mask detection (text-prompted, batched)
# ---------------------------------------------------------------------------


def detect_concept_masks(
    images,
    tag: str,
    detector: str,
    model,
    batch_size: int = 2,
    concept_masks_per_image: int = 5,
    return_scores: bool = False,
) -> List[List[Tuple[List[int], Any]]]:
    """Run text-prompted detection for *tag* and return top masks per image.

    Returns:
        List parallel to *images*.  Each element is a list of at most
        *concept_masks_per_image* ``(bbox_xywh, mask_np)`` tuples (sorted
        by area, largest first).
    """
    if detector == "langsam":
        pairs_per_img = predict_bboxes_and_masks_for_tag_batched(
            model,
            images,
            tag=tag,
            batch_size=batch_size,
        )
        if return_scores:
            pairs_per_img = [
                [(bbox, mask, float(_mask_area(mask))) for bbox, mask in pairs]
                for pairs in pairs_per_img
            ]
    else:
        raise ValueError(f"Unknown detector: {detector}")

    # Keep only top-K by area unless the caller requested scores for adaptive
    # selection.
    if return_scores:
        return pairs_per_img

    result = []
    for pairs in pairs_per_img:
        sorted_pairs = sorted(
            pairs, key=lambda p: _mask_area(p[1]), reverse=True
        )
        result.append(sorted_pairs[:concept_masks_per_image])
    return result


def _adaptive_concept_keep_count(
    scores: List[float],
) -> Tuple[int, Optional[float]]:
    """Concept-mask count picked from a histogram of the detection scores."""
    scores = [float(score) for score in scores if score is not None]
    if not scores:
        return 0, None
    if len(scores) == 1:
        return 1, scores[0]

    values = np.asarray(scores, dtype=float)
    bin_count = min(10, max(3, int(np.sqrt(len(values)))))
    hist, edges = np.histogram(values, bins=bin_count)
    peak_idx = int(np.argmax(hist))
    threshold = float(edges[peak_idx])
    keep_count = int(np.sum(values >= threshold))
    return max(1, keep_count), threshold


# ---------------------------------------------------------------------------
# Automatic (prompt-free) mask detection
# ---------------------------------------------------------------------------


def detect_auto_masks(
    images,
    detector: str,
    model,
    topn: int = 10,
    min_mask_area: int = 100,
) -> List[List[Tuple[List[int], Any]]]:
    """Run prompt-free segmentation.

    For LangSAM: uses SAM2AutomaticMaskGenerator.

    Returns:
        List parallel to *images*, each a list of ``(bbox_xywh, mask_np)``
        tuples.
    """
    if detector == "none":
        return [[] for _ in images]
    if detector == "langsam":
        return predict_all_masks_langsam(
            model, images, topn=topn, min_mask_area=min_mask_area
        )
    else:
        raise ValueError(f"Unknown detector: {detector}")


# ---------------------------------------------------------------------------
# Combine concept + auto masks for one image
# ---------------------------------------------------------------------------


def _build_masks_for_image(
    concept_pairs: List[Tuple[List[int], Any]],
    auto_pairs: List[Tuple[List[int], Any]],
    masks_per_image: int,
    concept_masks_per_image: int,
    min_mask_area: int = 100,
) -> Tuple[List[Tuple[List[int], Any]], int]:
    """Merge concept + auto masks, subtract concept from auto, enforce limits.

    The concept side is always collapsed into a single union mask.  The
    ``concept_masks_per_image`` value controls how many detected concept
    instances are allowed to contribute to that union, not how many concept
    masks are written out.

    When text-prompted concept detection returns nothing (common for abstract
    concepts like colours, textures, or "outdoor"), the largest auto mask is
    **promoted** to concept so that every image always has at least one
    ``is_concept=True`` mask (when ``concept_masks_per_image > 0``).

    Returns:
        ``(final_pairs, n_concept)`` where the first *n_concept* entry is the
        unioned concept mask (``is_concept=True``).
    """
    # --- Normalise polarity of auto masks ---------------------------------
    # An automatic mask can cover the background instead of an object (fg_ratio
    # scene (fg_ratio > 50%).  Invert those so the segment represents the
    # smaller, more distinctive region.  Recompute bbox after inversion.
    normalised_auto: List[Tuple[List[int], Any]] = []
    for bb, m in auto_pairs:
        if m is None:
            normalised_auto.append((bb, m))
            continue
        m_norm = _normalize_mask_polarity(m)
        if m_norm is not m:
            # Mask was inverted — recompute bbox from the new foreground
            ys, xs = np.where(m_norm)
            if len(ys) == 0:
                continue  # inversion produced empty mask
            bb = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            ]
        normalised_auto.append((bb, m_norm))
    auto_pairs = normalised_auto

    # --- concept masks (always first) ---
    if concept_masks_per_image > 0:
        concept_candidates = concept_pairs[:concept_masks_per_image]
    else:
        concept_candidates = []

    concept_union = None
    for _, cm in concept_candidates:
        if cm is None:
            continue
        if concept_union is None:
            concept_union = cm.copy()
        else:
            concept_union = concept_union | _resize_mask_like(cm, concept_union)

    kept_concept: List[Tuple[List[int], Any]] = []
    if concept_union is not None:
        ys, xs = np.where(concept_union)
        if len(ys) > 0:
            bbox_union = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            ]
            kept_concept = [(bbox_union, concept_union)]
    n_concept = len(kept_concept)

    # --- Fallback: promote largest auto candidate(s) when concept detection
    # missed ---
    if n_concept == 0 and concept_masks_per_image > 0 and len(auto_pairs) > 0:

        def _candidate_size(bb, m):
            if m is not None:
                return _mask_area(m)
            # Bbox-only candidates (detector="none" crop modes --
            # sliding_window,
            # random -- never carry a pixel mask by construction) fall back to
            # bbox area, so "always promote something" also works for them.
            # Without this, every auto_pair here has m=None, the old `m is not
            # None` filter drops all of them, and no image ever gets a
            # concept crop for these modes.
            return int(bb[2]) * int(bb[3])

        # Track by original index, not id(m): every bbox-only candidate
        # shares id(None), which would make the old id-based de-dup wrongly
        # match (and drop) every candidate once any one of them is promoted.
        indexed = [
            (idx, bb, m)
            for idx, (bb, m) in enumerate(auto_pairs)
            if _candidate_size(bb, m) >= min_mask_area
        ]
        indexed.sort(
            key=lambda item: _candidate_size(item[1], item[2]), reverse=True
        )
        if indexed and indexed[0][2] is None:
            # Bbox-only candidates aren't distinct detected object instances
            # (just arbitrary geometric windows), so promote only the single
            # largest one as a coarse "this represents the tag" proxy region
            # -- not up to concept_masks_per_image of them (often large, e.g.
            # 2000, which would tag almost every window as "concept").
            promote_count = 1
        else:
            promote_count = min(concept_masks_per_image, len(indexed))
        promoted = indexed[:promote_count]
        kept_concept = [(bb, m) for _, bb, m in promoted]
        n_concept = len(kept_concept)
        promoted_indices = {idx for idx, _, _ in promoted}
        auto_pairs = [
            p for i, p in enumerate(auto_pairs) if i not in promoted_indices
        ]

    # --- auto masks (subtract ALL previously placed masks so they are
    # non-overlapping) ---
    # Start the running union from concept masks; each accepted auto mask is
    # added too.
    placed_union = concept_union.copy() if concept_union is not None else None
    remaining_slots = (
        max(0, masks_per_image - n_concept)
        if masks_per_image > 0
        else len(auto_pairs)
    )
    kept_auto: List[Tuple[List[int], Any]] = []
    for bbox, mask in auto_pairs:
        if len(kept_auto) >= remaining_slots:
            break
        if mask is None:
            continue
        if placed_union is not None:
            trimmed = _subtract_mask(mask, placed_union)
            if _mask_area(trimmed) < min_mask_area:
                continue
            # Reject masks that became edge-like after subtraction
            if not _is_mask_solid(trimmed):
                continue
            # Update the running union with the trimmed (non-overlapping) mask
            placed_union = placed_union | _resize_mask_like(
                trimmed, placed_union
            )
            kept_auto.append((bbox, trimmed))
        else:
            if _mask_area(mask) >= min_mask_area:
                # First mask with no prior union — start one
                placed_union = mask.copy()
                kept_auto.append((bbox, mask))

    return kept_concept + kept_auto, n_concept


# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _record_for_image(
    tag_bucket: Dict[str, dict],
    rel_path: str,
    image_size: Tuple[int, int],
    patch_size: int,
    pairs: List[Tuple[List[int], Any]],
    n_concept: int,
    store_rle: bool = True,
    normalize_bbox: bool = True,
):
    """Serialize one image's masks into the tag bucket.

    normalize_bbox: re-center each bbox to a fixed patch_size x patch_size
    square, discarding its real width/height. Correct (and intentional) for
    sliding_window/random modes, which generate fixed-size geometric windows by
    design. Must be False for real-detector modes (langsam) -- otherwise
    the genuine detected object's bounding box (which varies per object, e.g. a
    knife vs. a cat) gets silently replaced with an arbitrary centered square,
    corrupting the crop geometry.
    """

    def _normalize_bbox_to_patch_size(
        bbox: List[int],
        image_wh: Tuple[int, int],
        target_patch: int,
    ) -> List[int]:
        """A fixed square patch centred on a bbox, clamped to the image."""
        iw, ih = int(image_wh[0]), int(image_wh[1])
        if iw <= 0 or ih <= 0:
            return [0, 0, 1, 1]

        x, y, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            bw, bh = 1, 1

        cx = x + (bw / 2.0)
        cy = y + (bh / 2.0)

        side = (
            int(target_patch) if int(target_patch) > 0 else max(1, min(iw, ih))
        )
        side_w = max(1, min(iw, side))
        side_h = max(1, min(ih, side))

        nx = int(round(cx - side_w / 2.0))
        ny = int(round(cy - side_h / 2.0))
        nx = max(0, min(nx, iw - side_w))
        ny = max(0, min(ny, ih - side_h))
        return [int(nx), int(ny), int(side_w), int(side_h)]

    def _clamp_bbox_to_image(
        bbox: List[int], image_wh: Tuple[int, int]
    ) -> List[int]:
        """Clamp a real bbox to image bounds without altering its size/shape."""
        iw, ih = int(image_wh[0]), int(image_wh[1])
        x, y, bw, bh = [int(v) for v in bbox]
        bw, bh = max(1, bw), max(1, bh)
        x = max(0, min(x, max(0, iw - bw)))
        y = max(0, min(y, max(0, ih - bh)))
        bw = min(bw, iw - x) if iw > 0 else bw
        bh = min(bh, ih - y) if ih > 0 else bh
        return [x, y, max(1, bw), max(1, bh)]

    w, h = image_size
    masks_list = []
    for i, (bbox, mask) in enumerate(pairs):
        if normalize_bbox:
            out_bbox = _normalize_bbox_to_patch_size(
                [int(v) for v in bbox],
                (w, h),
                patch_size,
            )
        else:
            out_bbox = _clamp_bbox_to_image([int(v) for v in bbox], (w, h))
        mask_entry = {
            "bbox": out_bbox,
            "is_concept": i < n_concept,
        }
        if store_rle and mask is not None:
            mask_entry["rle"] = _encode_mask_rle(mask)
        masks_list.append(mask_entry)

    tag_bucket[rel_path] = {
        "meta": {
            "image_size": [w, h],
            "patch_size": patch_size,
            "virtual_w": w,
            "virtual_h": h,
        },
        "masks_rle": masks_list,
    }
    if n_concept > 0:
        tag_bucket[rel_path]["meta"]["concept_mask_index"] = 0


def load_mapping(path: str) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, list)}


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_mapping(
    mapping_json: str,
    image_root: str,
    output_json: str,
    *,
    detector: str = "langsam",
    batch_size: int = 2,
    masks_per_image: int = 5,
    concept_masks_per_image: int = 1,
    min_images_per_tag: int = 5,
    max_images_per_tag: int = 0,
    image_size_width: Optional[int] = 512,
    patch_size: int = 200,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = True,
    positive_negative_segment: bool = False,
):
    """Process a tag->images mapping and write a crops JSON with RLE masks.

    For each tag and image:
      1. Produce ``concept_masks_per_image`` concept masks (text-prompted
         with the tag string).
      2. Fill remaining slots (up to ``masks_per_image``) with prompt-free
         automatic masks.
      3. Subtract concept mask area from auto masks so they don't overlap.

    When ``positive_negative_segment=True``:
      - All detected concept instances are unioned into a single FG mask.
      - An inverted copy becomes the "background" mask.
      - Exactly 2 masks per image: [concept (fg), inverted (bg)].
      - ``masks_per_image`` is forced to 2.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    detector_none = detector == "none"
    store_rle = os.environ.get("RLE", "1") != "0"
    clip_similarity_threshold = float(
        os.environ.get("CLIP_SIMILARITY_THRESHOLD", "0.5")
    )
    crop_mode = os.environ.get("CROP_MODE", "random").lower()
    sliding_window_stride_ratio = float(
        os.environ.get("SLIDING_WINDOW_STRIDE_RATIO", "0.3")
    )
    random_crop_per_image = int(os.environ.get("CROP_PER_IMAGE", "0"))
    if detector_none and crop_mode not in {"random", "sliding_window"}:
        # Detector-free mode supports CLIP-filtered random/sliding-window crops
        # only.
        crop_mode = "random"

    # Class-free segmentation (concept_masks_per_image=0): every mask comes
    # from the automatic mask generator; the tag text is never sent to the
    # detector and no mask slot is reserved for it.
    class_free_masks = not detector_none and concept_masks_per_image == 0
    if class_free_masks and positive_negative_segment:
        raise ValueError(
            "concept_masks_per_image=0 (class-free masks) cannot be "
            "combined with "
            "positive_negative_segment=1, which needs a tag mask as foreground."
        )
    # Sliding windows are ranked by CLIP similarity to the tag text unless
    # SLIDING_WINDOW_TAG_FILTER=0; near-duplicate removal (image-image CLIP
    # similarity) applies either way and uses no tag information.
    sliding_window_tag_filter = (
        os.environ.get("SLIDING_WINDOW_TAG_FILTER", "1") != "0"
    )

    # --- Override masks_per_image for binary segmentation mode ---
    # In pos/neg mode the output is always 2 masks: 1 fg (union of the
    # requested concept detections) + 1 bg (inverse).  The requested
    # concept count still controls how many detections we fetch and union.
    detect_topn_value: int = max(concept_masks_per_image, 1)
    adaptive_concept_masks = concept_masks_per_image < 0
    if adaptive_concept_masks:
        detect_topn_value = max(masks_per_image * 2, 20)
    if positive_negative_segment:
        masks_per_image = 2
        if not adaptive_concept_masks:
            detect_topn_value = max(concept_masks_per_image, 1)
        print(
            f"[positive_negative_segment] Binary fg/bg mode: "
            f"masks_per_image forced to {masks_per_image}, "
            f"concept_masks_per_image={concept_masks_per_image} "
            f"(detect_topn={detect_topn_value}, "
            "concept detections will be unioned)"
        )

    mapping = load_mapping(mapping_json)

    # --- Load model once ---
    if detector_none:
        model = None
    elif detector == "langsam":
        model = _load_langsam(device=device)
    else:
        raise ValueError(
            f"Unknown detector: {detector}. Use 'none' or 'langsam'."
        )

    clip_model = clip_preprocess = clip_device = None
    if detector_none:
        clip_model, clip_preprocess, clip_device = _load_clip_model(
            device=device
        )

    # Start fresh each run — no resume support for a clean, faster pipeline
    result: Dict[str, dict] = {}

    # --- Filter valid tags ---
    valid_tags: List[Tuple[str, List[str]]] = []
    for tag, rels in mapping.items():
        if not isinstance(rels, list) or len(rels) < min_images_per_tag:
            if verbose:
                print(
                    f"[skip] tag='{tag}': only "
                    f"{len(rels) if isinstance(rels, list) else 0} images "
                    f"(need {min_images_per_tag})"
                )
            continue
        if max_images_per_tag > 0 and len(rels) > max_images_per_tag:
            rels = random.sample(rels, max_images_per_tag)
        valid_tags.append((tag, rels))

    if not valid_tags:
        raise RuntimeError(
            f"No concept tag survived filtering: all {len(mapping)} tags in "
            f"'{mapping_json}' have fewer than min_images_per_tag="
            f"{min_images_per_tag} images. Lower MIN_IMAGES_PER_TAG or process "
            "more images in step 1 (IMAGE_BUDGET) — an empty crops.json would "
            "make every downstream step fail."
        )

    total_images = sum(len(rels) for _, rels in valid_tags)
    print(
        f"Processing {total_images} images across {len(valid_tags)} tags "
        f"(detector={detector}, masks_per_image={masks_per_image}, "
        f"concept={concept_masks_per_image})"
    )

    pbar = tqdm(
        total=total_images,
        desc="Segmenting",
        unit="img",
        disable=not show_progress,
        file=sys.stderr,
    )

    # Tag-level progress bar (concepts segmented / remaining)
    tag_pbar = tqdm(
        total=len(valid_tags),
        desc="Tags",
        unit="tag",
        disable=not show_progress,
        file=sys.stderr,
        leave=True,
    )

    # Progress / ETA tracking
    start_time = time.time()
    processed_images = 0
    status_log_path = os.path.join(
        os.path.dirname(output_json) or ".", "crop_status.log"
    )
    status_write_every = int(os.environ.get("STATUS_WRITE_EVERY", "10"))

    def _write_status(force: bool = False, current_tag: Optional[str] = None):
        if not force and (
            status_write_every <= 0
            or (processed_images % status_write_every) != 0
        ):
            return
        elapsed = time.time() - start_time
        rate = (processed_images / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, total_images - processed_images)
        eta = (remaining / rate) if rate > 0 else None
        eta_s = f"{int(eta)}s" if eta is not None else "N/A"
        tag_str = current_tag[:20] if current_tag else ""
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"processed={processed_images}/{total_images} "
            f"elapsed={elapsed:.1f}s rate={rate:.2f}img/s eta={eta_s} "
            f"tag={tag_str}"
        )
        if pbar:
            pbar.set_postfix(eta=eta_s)
            tqdm.write(line)
        else:
            print(line)
        try:
            with open(status_log_path, "a", encoding="utf-8") as sf:
                sf.write(line + "\n")
        except OSError as e:  # the status file is a convenience only
            print(f"Warning: could not append to {status_log_path}: {e}")

    for tag_idx, (tag, rels) in enumerate(valid_tags):
        if tag_pbar:
            tag_pbar.set_postfix(
                current=tag[:20], remaining=len(valid_tags) - tag_idx
            )
        tag_bucket: Dict[str, dict] = result.setdefault(tag, {})

        # Resolve absolute paths and filter already-done
        todo: List[Tuple[str, str]] = []  # (rel, abs)
        # This block of code is for creating segmentation if there is a re-run
        # in new data, if there are concepts exist in old data then it will not
        # create segmentation for that image but if concept is not exist in old
        # data then it will create segmentation for that image.
        for rel in rels:
            abs_p = os.path.join(image_root, rel)
            if not os.path.isfile(abs_p):
                continue
            # Skip only if this image was already processed AND has valid
            # concept masks (when concept detection is requested).  This
            # prevents stale entries from a broken/partial earlier run from
            # being silently kept.
            if rel in tag_bucket:
                existing = tag_bucket[rel]
                has_concept = any(
                    m.get("is_concept", False)
                    for m in existing.get("masks_rle", [])
                )
                if has_concept or concept_masks_per_image == 0:
                    if pbar:
                        pbar.update(1)
                    processed_images += 1
                    _write_status(current_tag=tag)
                    continue
                # Stale entry without concept masks → re-process
                del tag_bucket[rel]
            todo.append((rel, abs_p))

        if not todo:
            continue

        # Process in small chunks to limit peak GPU/CPU memory.
        # Each chunk: load images → concept detect → auto detect → combine →
        # free.
        chunk_size = max(batch_size, 2)  # process this many images at a time

        for chunk_start in range(0, len(todo), chunk_size):
            chunk = todo[chunk_start : chunk_start + chunk_size]

            # --- Load & resize images for this chunk ---
            pil_images = []
            sizes = []
            for rel, abs_p in chunk:
                img, (w, h), _ = _load_and_resize(abs_p, image_size_width)
                pil_images.append(img)
                sizes.append((w, h))

            # --- 1. Concept masks (text-prompted, batched) ---
            # In pos/neg mode _detect_topn > concept_masks_per_image so
            # all instances are returned for the union step.
            if detector_none or class_free_masks:
                concept_per_img = [[] for _ in pil_images]
            else:
                concept_per_img = detect_concept_masks(
                    pil_images,
                    tag=tag,
                    detector=detector,
                    model=model,
                    batch_size=batch_size,
                    concept_masks_per_image=detect_topn_value,
                    return_scores=adaptive_concept_masks,
                )

            _cleanup_gpu()
            # --- 2. Auto masks or binary fg/bg split ---
            if detector_none:
                random_count = masks_per_image if masks_per_image > 0 else 1
                for i, (rel, _) in enumerate(chunk):
                    if crop_mode == "sliding_window":
                        pairs = _build_sliding_window_pairs_for_image(
                            pil_images[i],
                            sizes[i],
                            random_count,
                            clip_model,
                            clip_preprocess,
                            clip_device,
                            similarity_threshold=clip_similarity_threshold,
                            window_size=patch_size,
                            stride_ratio=sliding_window_stride_ratio,
                            tag=tag if sliding_window_tag_filter else None,
                        )
                    else:
                        pairs = _build_diverse_random_pairs_for_image(
                            pil_images[i],
                            sizes[i],
                            random_count,
                            random_crop_per_image
                            if random_crop_per_image > 0
                            else None,
                            patch_size,
                            random,
                            clip_model,
                            clip_preprocess,
                            clip_device,
                            similarity_threshold=clip_similarity_threshold,
                        )
                    _record_for_image(
                        tag_bucket,
                        rel,
                        sizes[i],
                        patch_size,
                        pairs,
                        0,
                        store_rle=store_rle,
                    )
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(
                            tag=tag[:12],
                            masks=len(pairs),
                            concept=0,
                            mode=crop_mode,
                        )
                    processed_images += 1
                    _write_status(current_tag=tag)

            elif positive_negative_segment:
                # Binary mode: union all concept instances into one mask,
                # then create an inverted copy as the "background" mask.
                # Dilation/erosion via MASK_BOUNDARY_PIXELS is applied
                # later at inference time (save_features.py) so the
                # stored masks stay canonical.
                auto_per_img = [[] for _ in pil_images]  # unused

                for i, (rel, _) in enumerate(chunk):
                    cp_raw = (
                        concept_per_img[i] if i < len(concept_per_img) else []
                    )
                    if adaptive_concept_masks:
                        cp_scores = [
                            score for _, _, score in cp_raw if score is not None
                        ]
                        keep_count, threshold = _adaptive_concept_keep_count(
                            cp_scores
                        )
                        if threshold is None:
                            cp = []
                        else:
                            cp = [
                                (bbox, mask)
                                for bbox, mask, score in cp_raw
                                if score is None or score >= threshold
                            ][:keep_count]
                    else:
                        cp = cp_raw[: max(concept_masks_per_image, 0)]

                    if len(cp) == 0:
                        # No concept mask detected — record empty
                        _record_for_image(
                            tag_bucket,
                            rel,
                            sizes[i],
                            patch_size,
                            [],
                            0,
                            store_rle=store_rle,
                            normalize_bbox=False,
                        )
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix(
                                tag=tag[:12], masks=0, concept=0, mode="pos_neg"
                            )
                        processed_images += 1
                        _write_status(current_tag=tag)
                        continue

                    # --- Union all detected concept instances into one mask ---
                    union_mask = None
                    for _, mask_c in cp:
                        if mask_c is None:
                            continue
                        if union_mask is None:
                            union_mask = mask_c.copy()
                        else:
                            # Instances may come at different resolutions.
                            union_mask = union_mask | _resize_mask_like(
                                mask_c, union_mask
                            )

                    if union_mask is None:
                        _record_for_image(
                            tag_bucket,
                            rel,
                            sizes[i],
                            patch_size,
                            [],
                            0,
                            store_rle=store_rle,
                            normalize_bbox=False,
                        )
                        if pbar:
                            pbar.update(1)
                        processed_images += 1
                        _write_status(current_tag=tag)
                        continue

                    mask_fg = union_mask

                    # Compute FG bbox from union mask
                    ys_fg, xs_fg = np.where(mask_fg)
                    if len(ys_fg) == 0:
                        _record_for_image(
                            tag_bucket,
                            rel,
                            sizes[i],
                            patch_size,
                            [],
                            0,
                            store_rle=store_rle,
                            normalize_bbox=False,
                        )
                        if pbar:
                            pbar.update(1)
                        processed_images += 1
                        _write_status(current_tag=tag)
                        continue
                    bbox_fg = [
                        int(xs_fg.min()),
                        int(ys_fg.min()),
                        int(xs_fg.max() - xs_fg.min() + 1),
                        int(ys_fg.max() - ys_fg.min() + 1),
                    ]

                    # Create inverted mask (background)
                    mask_bg = ~mask_fg
                    ys_bg, xs_bg = np.where(mask_bg)
                    if len(ys_bg) == 0:
                        # Concept mask covers entire image — only fg
                        pairs = [(bbox_fg, mask_fg)]
                        n_concept = 1
                    else:
                        bbox_bg = [
                            int(xs_bg.min()),
                            int(ys_bg.min()),
                            int(xs_bg.max() - xs_bg.min() + 1),
                            int(ys_bg.max() - ys_bg.min() + 1),
                        ]
                        # pairs: [foreground (concept), background (inverted)]
                        pairs = [(bbox_fg, mask_fg), (bbox_bg, mask_bg)]
                        n_concept = 1

                    _record_for_image(
                        tag_bucket,
                        rel,
                        sizes[i],
                        patch_size,
                        pairs,
                        n_concept,
                        store_rle=store_rle,
                        normalize_bbox=False,
                    )

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(
                            tag=tag[:12],
                            masks=len(pairs),
                            concept=n_concept,
                            mode="pos_neg",
                        )
                    processed_images += 1
                    _write_status(current_tag=tag)
            else:
                # --- Standard multi-mask mode ---
                # One slot is reserved for the unioned tag mask, except in
                # class-free mode where every slot holds an automatic mask.
                auto_topn = (
                    max(masks_per_image, 1)
                    if class_free_masks
                    else max(masks_per_image - 1, 1)
                )
                auto_per_img = detect_auto_masks(
                    pil_images,
                    detector=detector,
                    model=model,
                    topn=auto_topn,
                    min_mask_area=100,
                )

                _cleanup_gpu()

                # --- 3. Combine, subtract, record ---
                for i, (rel, _) in enumerate(chunk):
                    cp_raw = (
                        concept_per_img[i] if i < len(concept_per_img) else []
                    )
                    if adaptive_concept_masks:
                        cp_scores = [
                            score for _, _, score in cp_raw if score is not None
                        ]
                        keep_count, threshold = _adaptive_concept_keep_count(
                            cp_scores
                        )
                        if threshold is None:
                            cp = []
                        else:
                            cp = [
                                (bbox, mask)
                                for bbox, mask, score in cp_raw
                                if score is None or score >= threshold
                            ][:keep_count]
                    else:
                        cp = cp_raw[: max(concept_masks_per_image, 0)]
                    ap = auto_per_img[i] if i < len(auto_per_img) else []

                    pairs, n_concept = _build_masks_for_image(
                        cp,
                        ap,
                        masks_per_image=masks_per_image,
                        concept_masks_per_image=len(cp),
                    )

                    _record_for_image(
                        tag_bucket,
                        rel,
                        sizes[i],
                        patch_size,
                        pairs,
                        n_concept,
                        store_rle=store_rle,
                        normalize_bbox=False,
                    )

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(
                            tag=tag[:12], masks=len(pairs), concept=n_concept
                        )
                    processed_images += 1
                    _write_status(current_tag=tag)

            # Free chunk-local references before the next chunk.
            pil_images = []
            concept_per_img = []
            auto_per_img = []
            _cleanup_gpu()

        # Do not flush after each tag (no resume support) — continue
        # accumulating
        gc.collect()

        if tag_pbar:
            tag_pbar.update(1)
        _write_status(force=True, current_tag=tag)

    if pbar:
        pbar.close()
    if tag_pbar:
        tag_pbar.close()

    # Write final output once (no per-tag flush/resume support)
    _atomic_write_json(output_json, result)

    # Final summary
    total_tags_out = len(result)
    total_imgs_out = sum(len(v) for v in result.values() if isinstance(v, dict))
    print(
        f"Wrote {total_imgs_out} images across {total_tags_out} tags -> "
        f"{output_json}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    apply_determinism_setting()  # DETERMINISTIC=1
    parser = argparse.ArgumentParser(
        description="Build mask-centric crops JSON from a tag->image mapping.",
    )
    parser.add_argument(
        "--mapping_json",
        required=True,
        help="JSON file: {tag: [relative_paths...]}",
    )
    parser.add_argument(
        "--image_root",
        required=True,
        help="Root directory that relative paths are resolved against",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to write the output crops JSON",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="langsam",
        choices=["none", "langsam"],
        help="Segmentation model (default: langsam)",
    )
    parser.add_argument(
        "--masks_per_image",
        type=int,
        default=5,
        help="Total masks per image (concept + auto)",
    )
    parser.add_argument(
        "--concept_masks_per_image",
        type=int,
        default=5,
        help="Number of concept (text-prompted) masks per image",
    )
    parser.add_argument(
        "--min_images_per_tag",
        type=int,
        default=5,
        help="Skip tags with fewer images than this",
    )
    parser.add_argument(
        "--max_images_per_tag",
        type=int,
        default=0,
        help="Cap images per tag (0 = no cap)",
    )
    parser.add_argument(
        "--image_size_width",
        type=int,
        default=512,
        help="Resize width for detection (height auto by aspect ratio)",
    )
    parser.add_argument(
        "--patch_size", type=int, default=200, help="Patch size metadata"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for text-prompted detection",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cpu, cuda, or cuda:N"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars"
    )
    parser.add_argument(
        "--positive_negative_segment",
        type=int,
        default=0,
        help="1 = binary fg/bg masks (concept + inverted), "
        "0 = multi-mask pipeline (default: 0, reads "
        "POSITIVE_NEGATIVE_SEGMENT env var as fallback)",
    )

    args = parser.parse_args()

    # Resolve positive_negative_segment: CLI wins, then env var
    pos_neg = args.positive_negative_segment
    if pos_neg == 0:
        pos_neg = int(
            os.environ.get(
                "POSITIVE_NEGATIVE_SEGMENT",
                os.environ.get("POSITIVE_NEGATVE_SEGMENT", "0"),
            )
        )

    process_mapping(
        mapping_json=args.mapping_json,
        image_root=args.image_root,
        output_json=args.output_json,
        detector=args.detector,
        batch_size=args.batch_size,
        masks_per_image=args.masks_per_image,
        concept_masks_per_image=args.concept_masks_per_image,
        min_images_per_tag=args.min_images_per_tag,
        max_images_per_tag=args.max_images_per_tag,
        image_size_width=args.image_size_width,
        patch_size=args.patch_size,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
        show_progress=not getattr(args, "no_progress", False),
        positive_negative_segment=bool(pos_neg),
    )


if __name__ == "__main__":
    main()
