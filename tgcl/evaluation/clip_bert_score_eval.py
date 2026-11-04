#!/usr/bin/env python3
"""Retrieval scores of explanations (step 7, the paper's retrieval table).

For every answer word, the top-k concepts of the explanation are compared with
the word: BERTScore against the concepts' class names and CLIPScore against
their grounding crops, next to the same scores for random concepts.
"""

import argparse
import ast
import csv
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from bert_score import BERTScorer
import clip
import numpy as np
from PIL import Image
import torch
from transformers import logging as hf_logging

from tgcl.device_utils import get_device_config
from tgcl.helpers.explanations import expand
from tgcl.helpers.grounding import DEFAULT_IMAGE_WIDTH
from tgcl.helpers.grounding import grounding_crop
from tgcl.helpers.image_refs import load_image
from tgcl.helpers.utils import apply_determinism_setting
from tgcl.models.revisions import BERTSCORE_LAYERS
from tgcl.models.revisions import BERTSCORE_MODEL
from tgcl.models.revisions import pinned_snapshot

hf_logging.set_verbosity_error()


_PUNCT_RE = re.compile(r"[\.,;:!?()\[\]{}\"']")
_SPACE_RE = re.compile(r"\s+")

_CLIP_MODEL = None
_CLIP_PREPROCESS = None


def normalize_piece(piece: str) -> str:
    piece = (piece or "").lower()
    for prefix in ("##", "▁", "Ġ"):
        if piece.startswith(prefix):
            piece = piece[len(prefix) :]
    return piece


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text


def _split_text(text: str) -> List[str]:
    text = _normalize_text(text)
    return text.split() if text else []


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [value]
        except (ValueError, SyntaxError):  # a plain string, not a literal
            return [value]
    if isinstance(value, list):
        return value
    return [value]


def build_gt_token_map(
    gt_words: List[str], gt_tokens: List[str], sim_thresh: float = 0.8
) -> List[str]:
    """The answer word each answer token belongs to.

    Tokens are joined left to right until they spell the next word (exactly,
    or with SequenceMatcher ratio >= sim_thresh once long enough).

    Args:
        gt_words: The answer's words, in order.
        gt_tokens: Its tokens, in order.
        sim_thresh: Similarity that counts as a match.

    Returns:
        One word per token (tokens past the last word map to it).
    """
    if not gt_words:
        return [""] * len(gt_tokens)

    words = [w.lower() for w in gt_words]
    tokens = [normalize_piece(t) for t in gt_tokens]

    out: List[str] = []
    i = 0
    j = 0
    while j < len(tokens):
        if i >= len(words):
            out.append(gt_words[-1])
            j += 1
            continue

        target = words[i]
        buf = ""
        start_j = j

        while j < len(tokens) and len(buf) <= len(target):
            buf += tokens[j]
            j += 1

            if buf == target:
                out.extend([gt_words[i]] * (j - start_j))
                i += 1
                break

            if len(buf) >= len(target):
                sim = SequenceMatcher(None, buf, target).ratio()
                if sim >= sim_thresh:
                    out.extend([gt_words[i]] * (j - start_j))
                    i += 1
                    break
        else:
            out.append(gt_words[i])
            j = start_j + 1

    return out


# Repository root (tgcl/evaluation/<this file>): relative image paths resolve
# against it.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Width crops_to_json resized the training images to (IMAGE_SIZE_WIDTH):
# grounding boxes are in that image. Set from --grounding_image_width in main().
GROUNDING_IMAGE_WIDTH = int(
    os.environ.get("IMAGE_SIZE_WIDTH", DEFAULT_IMAGE_WIDTH)
)


def _resolve_image(path: str) -> Path:
    """Grounding image path as stored (absolute, or relative to the cwd or the
    repo root)."""
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    return REPO_ROOT / p


def _grounding_crop(path: str, bbox: Any) -> Image.Image:
    """Grounding segment as step 4 saw it: image resized to
    GROUNDING_IMAGE_WIDTH, box [x, y, w, h]."""
    resolved = _resolve_image(path)
    if not resolved.exists():
        # A missing crop used to be skipped silently (CLIP 0 when all were
        # missing).
        raise FileNotFoundError(
            f"grounding image not found: {path} (resolved {resolved})"
        )
    return grounding_crop(resolved, bbox, GROUNDING_IMAGE_WIDTH)


class CLIPImageDataset(torch.utils.data.Dataset):
    """Images (PIL images or references) preprocessed for CLIP."""

    def __init__(self, images: List[Any], preprocess: Any) -> None:
        self.images = images
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = load_image(image)
        return {"image": self.preprocess(image)}


_BERT_SCORERS: Dict[str, BERTScorer] = {}


def _bert_scorer(device: str) -> BERTScorer:
    """One BERTScorer per device. bert_score.score() rebuilds roberta-large on
    every call (~1 s), i.e. thousands of reloads per table; same defaults
    (lang="en": roberta-large, its default layer, no idf, no rescaling)."""
    if device not in _BERT_SCORERS:
        _BERT_SCORERS[device] = BERTScorer(
            model_type=pinned_snapshot(BERTSCORE_MODEL),
            num_layers=BERTSCORE_LAYERS,
            lang="en",
            device=device,
        )
    return _BERT_SCORERS[device]


def _load_clip_once(
    model_name: str = "ViT-B/32", device: str = "cuda"
) -> Tuple[Any, Any]:
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None or _CLIP_PREPROCESS is None:
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(
            model_name, device=device, jit=False
        )
        _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _extract_image_features(
    images: List[Any],
    model: Any,
    device: str,
    preprocess: Any,
    batch_size: int = 64,
) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
        CLIPImageDataset(images, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(str(device).startswith("cuda")),
    )
    image_features = []
    with torch.no_grad():
        for batch in loader:
            image_tensor = batch["image"].to(device)
            if str(device).startswith("cuda"):
                image_tensor = image_tensor.to(torch.float16)
            image_features.append(
                model.encode_image(image_tensor).cpu().numpy()
            )
    return np.vstack(image_features)


def _extract_text_features(
    candidates: List[str], model: Any, device: str, batch_size: int = 256
) -> np.ndarray:
    text_features = []
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            chunk = candidates[i : i + batch_size]
            tokenized = clip.tokenize(chunk, truncate=True).to(device)
            feats = model.encode_text(tokenized).cpu().numpy()
            text_features.append(feats)
    return np.vstack(text_features)


_CLIP_IMAGE_FEATURES: Dict[Tuple[str, str], np.ndarray] = {}
_CLIP_TEXT_FEATURES: Dict[str, np.ndarray] = {}


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.sum(x**2, axis=-1, keepdims=True))


def _clip_image_feature(
    path: str, bbox: Any, model: Any, device: str, preprocess: Any
) -> np.ndarray:
    """Unit CLIP feature of one grounding crop.

    Memoized: grounding crops repeat for every sample and k (a few dozen
    distinct crops per concept bank versus thousands of lookups)."""
    key = (str(path), repr(bbox))
    if key not in _CLIP_IMAGE_FEATURES:
        image = _grounding_crop(path, bbox)
        _CLIP_IMAGE_FEATURES[key] = _unit(
            _extract_image_features([image], model, device, preprocess)[0]
        )
    return _CLIP_IMAGE_FEATURES[key]


def _clip_text_feature(text: str, model: Any, device: str) -> np.ndarray:
    if text not in _CLIP_TEXT_FEATURES:
        _CLIP_TEXT_FEATURES[text] = _unit(
            _extract_text_features([text], model, device)[0]
        )
    return _CLIP_TEXT_FEATURES[text]


def _extract_topk_predictions(
    result: Dict[str, Any],
    ks: Tuple[int, ...],
    ignore_words: frozenset = frozenset(),
) -> Dict[str, Any]:
    """Per answer token: its answer word and the top-k retrieved concepts.

    Tokens whose answer word is in `ignore_words` (normalized, e.g. the binary
    prompt's "no"/"yes") are dropped from scoring.
    """
    gt_words = _split_text(result.get("model_output", ""))
    per_tokens = result.get("per_token_concepts", []) or []
    token_text_ranked = [t.get("token_text", "") for t in per_tokens]
    gt_token_word_map = build_gt_token_map(gt_words, token_text_ranked)

    top_concepts_per_token = [t.get("top_concepts", []) for t in per_tokens]
    # Empty words (no answer text) and ignore_words are not scored.
    keep = [
        i
        for i, word in enumerate(gt_token_word_map)
        if _normalize_text(word) and _normalize_text(word) not in ignore_words
    ]
    gt_token_word_map = [gt_token_word_map[i] for i in keep]
    top_concepts_per_token = [top_concepts_per_token[i] for i in keep]

    predictions_text = []
    predictions_images = []
    predictions_bboxes = []

    for pred_list in top_concepts_per_token:
        pred_text_k: Dict[int, List[str]] = {}
        pred_img_k: Dict[int, List[str]] = {}
        pred_bbox_k: Dict[int, List[Any]] = {}

        for k in ks:
            pred_k = pred_list[:k]
            text_items: List[str] = []
            image_items: List[str] = []
            bbox_items: List[Any] = []

            for item in pred_k:
                # concept_name: the clean, ground-truth category tag assigned
                # during vocab-filtering/purity-check (e.g. "apple") -- used
                # in place of text_grounding (a noisy logit-lens readout of
                # the concept vector, e.g. "light"/"fruit"/"lemon" for an
                # apple concept), since it's a far more reliable label for
                # comparing against the model's real generated word.
                for tg in _to_list(item.get("concept_name", [])):
                    norm = _normalize_text(str(tg))
                    if norm:
                        text_items.append(norm)

                img_list = _to_list(item.get("image_grounding_path", []))
                box_list = _to_list(item.get("image_grounding_bboxes", []))

                if len(box_list) == len(img_list):
                    for img_path, bbox in zip(img_list, box_list):
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(bbox)
                elif len(box_list) == 1 and len(img_list) > 1:
                    for img_path in img_list:
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(box_list[0])
                else:
                    for img_path in img_list:
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(None)

            pred_text_k[k] = sorted(set(text_items))
            pred_img_k[k] = image_items
            pred_bbox_k[k] = bbox_items

        predictions_text.append(pred_text_k)
        predictions_images.append(pred_img_k)
        predictions_bboxes.append(pred_bbox_k)

    return {
        "gt_token_word_map": gt_token_word_map,
        "predictions_text": predictions_text,
        "prediction_image": predictions_images,
        "prediction_bboxes": predictions_bboxes,
    }


def _build_concept_bank(concept_path: Path) -> List[Dict[str, Any]]:
    concept_data = torch.load(str(concept_path), map_location="cpu")
    # concept_names: clean ground-truth category tag per concept (see
    # _extract_topk_predictions for why this replaced text_grounding).
    concept_names = concept_data.get("concept_names", [])
    image_paths = concept_data.get("image_grounding_paths", [])
    image_bboxes = concept_data.get("image_grounding_bboxes", [])

    num_concepts = max(
        len(concept_names) if isinstance(concept_names, list) else 0,
        len(image_paths) if isinstance(image_paths, list) else 0,
    )

    bank: List[Dict[str, Any]] = []
    for i in range(num_concepts):
        cn = concept_names[i] if i < len(concept_names) else []
        ip = image_paths[i] if i < len(image_paths) else []
        ib = image_bboxes[i] if i < len(image_bboxes) else []
        bank.append(
            {
                "concept_name": _to_list(cn),
                "image_grounding_paths": _to_list(ip),
                "image_grounding_bboxes": _to_list(ib),
            }
        )
    return bank


def _sample_indices(num_concepts: int, k: int, rng: random.Random) -> List[int]:
    if num_concepts <= 0 or k <= 0:
        return []
    if k <= num_concepts:
        return rng.sample(range(num_concepts), k)
    picked = list(range(num_concepts))
    while len(picked) < k:
        picked.append(rng.randrange(num_concepts))
    return picked


def _build_random_predictions(
    gt_token_word_map: List[str],
    concept_bank: List[Dict[str, Any]],
    ks: Tuple[int, ...],
    rng: random.Random,
) -> Dict[str, Any]:
    max_k = max(ks)
    num_concepts = len(concept_bank)

    predictions_text = []
    predictions_images = []
    predictions_bboxes = []

    for _ in gt_token_word_map:
        sampled = _sample_indices(num_concepts=num_concepts, k=max_k, rng=rng)

        pred_text_k: Dict[int, List[str]] = {}
        pred_img_k: Dict[int, List[str]] = {}
        pred_bbox_k: Dict[int, List[Any]] = {}

        for k in ks:
            chosen = sampled[:k]
            text_items: List[str] = []
            image_items: List[str] = []
            bbox_items: List[Any] = []

            for ci in chosen:
                concept_item = concept_bank[ci]
                for tg in concept_item["concept_name"]:
                    norm = _normalize_text(str(tg))
                    if norm:
                        text_items.append(norm)

                imgs = concept_item["image_grounding_paths"]
                bboxes = concept_item["image_grounding_bboxes"]

                if len(bboxes) == len(imgs):
                    for img_path, bbox in zip(imgs, bboxes):
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(bbox)
                elif len(bboxes) == 1 and len(imgs) > 1:
                    for img_path in imgs:
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(bboxes[0])
                else:
                    for img_path in imgs:
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(None)

            pred_text_k[k] = sorted(set(text_items))
            pred_img_k[k] = image_items
            pred_bbox_k[k] = bbox_items

        predictions_text.append(pred_text_k)
        predictions_images.append(pred_img_k)
        predictions_bboxes.append(pred_bbox_k)

    return {
        "gt_token_word_map": gt_token_word_map,
        "predictions_text": predictions_text,
        "prediction_image": predictions_images,
        "prediction_bboxes": predictions_bboxes,
    }


_BERT_PAIR_F1: Dict[Tuple[str, str], float] = {}


def _score_bert_pairs(
    pairs: List[Tuple[str, str]], device: str, batch_size: int = 256
) -> None:
    """BERTScore F1 of each new (candidate, reference) pair, in one batched
    call. Answer words and concept names repeat across samples and k, so each
    unique pair is scored once (idf off: a pair's score does not depend on its
    batch).
    """
    todo = sorted({p for p in pairs if p not in _BERT_PAIR_F1})
    if not todo:
        return
    _, _, f1 = _bert_scorer(device).score(
        [c for c, _ in todo],
        [r for _, r in todo],
        verbose=False,
        batch_size=batch_size,
    )
    _BERT_PAIR_F1.update(zip(todo, (float(x) for x in f1)))


def _sample_bertscore_per_k(
    gt_words: List[str],
    pred_text_list: List[Dict[int, List[str]]],
    k: int,
    bert_device: str,
) -> Optional[float]:
    """Mean over answer words of the mean F1 between the word and its top-k
    concept names (0 for a word without concepts); None if the sample has no
    scoreable word.
    """
    if not gt_words:
        return None
    pairs = [
        (pred, gt)
        for gt, pred_k in zip(gt_words, pred_text_list)
        for pred in pred_k.get(k, [])
    ]
    _score_bert_pairs(pairs, bert_device)
    per_word = [
        float(
            np.mean([_BERT_PAIR_F1[(pred, gt)] for pred in pred_k.get(k, [])])
        )
        if pred_k.get(k)
        else 0.0
        for gt, pred_k in zip(gt_words, pred_text_list)
    ]
    return float(np.mean(per_word))


def _sample_clipscore_per_k(
    gt_words: List[str],
    pred_img_list: List[Dict[int, List[str]]],
    pred_bbox_list: List[Dict[int, List[Any]]],
    k: int,
    clip_device: str,
) -> Optional[float]:
    """Mean over answer words of the mean CLIPScore (2.5 * max(cos, 0)) between
    the word and the grounding crops of its top-k concepts (0 for a word without
    crops); None if the sample has no scoreable word."""
    if not gt_words:
        return None

    model, preprocess = _load_clip_once(device=clip_device)

    w = 2.5  # CLIPScore weight
    mean_per_gt: List[float] = []
    for gt, pred_imgs_k, pred_bboxes_k in zip(
        gt_words, pred_img_list, pred_bbox_list
    ):
        imgs = list(pred_imgs_k.get(k, []))
        bboxes = (
            list(pred_bboxes_k.get(k, []))
            if isinstance(pred_bboxes_k, dict)
            else []
        )
        if len(bboxes) == len(imgs):
            pairs = zip(imgs, bboxes)
        elif len(bboxes) == 1 and len(imgs) > 1:
            pairs = ((img_path, bboxes[0]) for img_path in imgs)
        else:
            pairs = ((img_path, None) for img_path in imgs)
        feats = [
            _clip_image_feature(p, b, model, clip_device, preprocess)
            for p, b in pairs
        ]
        if not feats:
            mean_per_gt.append(0.0)
            continue
        text = _clip_text_feature(gt, model, clip_device)
        mean_per_gt.append(
            float(
                np.mean([w * max(float(np.dot(f, text)), 0.0) for f in feats])
            )
        )

    return float(np.mean(mean_per_gt)) if mean_per_gt else 0.0


def _per_sample_scores(
    records: List[Dict[str, Any]],
    ks: Tuple[int, ...],
    bert_device: str,
    clip_device: str,
) -> Dict[int, Dict[str, List[Optional[float]]]]:
    """{k: {"bert": [one score per record, None if unscoreable], "clip":
    [...]}}."""
    scores = {k: {"bert": [], "clip": []} for k in ks}
    # Every BERT pair of every record and k in one batched pass (fast path).
    _score_bert_pairs(
        [
            (pred, gt)
            for record in records
            for gt, pred_k in zip(
                record["gt_token_word_map"], record["predictions_text"]
            )
            for k in ks
            for pred in pred_k.get(k, [])
        ],
        bert_device,
    )
    for record in records:
        gt_words = record["gt_token_word_map"]
        pred_text_list = record["predictions_text"]
        pred_img_list = record["prediction_image"]
        pred_bbox_list = record["prediction_bboxes"]
        for k in ks:
            scores[k]["bert"].append(
                _sample_bertscore_per_k(
                    gt_words, pred_text_list, k, bert_device
                )
            )
            scores[k]["clip"].append(
                _sample_clipscore_per_k(
                    gt_words, pred_img_list, pred_bbox_list, k, clip_device
                )
            )
    return scores


def _summarize(
    scores: Dict[int, Dict[str, List[float]]],
) -> Dict[int, Dict[str, float]]:
    summary: Dict[int, Dict[str, float]] = {}
    for k, per_metric in scores.items():
        # Samples without a scoreable answer word (None) are not averaged.
        bvals = [v for v in per_metric["bert"] if v is not None]
        cvals = [v for v in per_metric["clip"] if v is not None]
        summary[k] = {
            "bert_mean": float(np.mean(bvals)) if bvals else 0.0,
            "bert_std": float(np.std(bvals)) if bvals else 0.0,
            "clip_mean": float(np.mean(cvals)) if cvals else 0.0,
            "clip_std": float(np.std(cvals)) if cvals else 0.0,
        }
    return summary


def write_per_sample_scores(
    path: Path,
    results: List[Dict[str, Any]],
    ranked: Dict[int, Dict[str, List[float]]],
    random_: Dict[int, Dict[str, List[float]]],
) -> Path:
    """One row per (sample, k): the scores the summary table averages."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "image_path",
                "rank_k",
                "bert",
                "clip",
                "random_bert",
                "random_clip",
            ]
        )
        for k in sorted(ranked):
            for i, result in enumerate(results):
                writer.writerow(
                    [i, result.get("image_path", ""), k]
                    + [
                        "" if v is None else v
                        for v in (
                            ranked[k]["bert"][i],
                            ranked[k]["clip"][i],
                            random_[k]["bert"][i],
                            random_[k]["clip"][i],
                        )
                    ]
                )
    return path


def main() -> None:
    apply_determinism_setting()  # DETERMINISTIC=1
    global GROUNDING_IMAGE_WIDTH
    parser = argparse.ArgumentParser(
        description="Evaluate BERT/CLIP top-k scores and random activation "
        "baseline."
    )
    parser.add_argument(
        "--json_path", required=True, help="Path to vlm_explanations.json"
    )
    parser.add_argument(
        "--concept_path",
        required=True,
        help="Path to combined_concept_*_raw.pth",
    )
    parser.add_argument(
        "--max_k", type=int, default=5, help="Evaluate k=1..max_k"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for baseline concept sampling",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for summary table"
    )
    parser.add_argument(
        "--output_prefix", default="clip_bert_topk", help="Output file prefix"
    )
    parser.add_argument(
        "--bert_device",
        default=os.environ.get("BERT_DEVICE", os.environ.get("DEVICE", "cuda")),
    )
    parser.add_argument(
        "--clip_device",
        default=os.environ.get("CLIP_DEVICE", os.environ.get("DEVICE", "cuda")),
    )
    parser.add_argument(
        "--ignore_words",
        default="",
        help="Comma-separated answer words not scored (e.g. 'no,yes' for the "
        "binary prompt).",
    )
    parser.add_argument(
        "--grounding_image_width",
        type=int,
        default=GROUNDING_IMAGE_WIDTH,
        help="IMAGE_SIZE_WIDTH of step 2: grounding boxes are in images "
        "resized to this width",
    )
    args = parser.parse_args()
    GROUNDING_IMAGE_WIDTH = args.grounding_image_width

    # Normalize device specs ("auto", "cuda:0,2", "cuda:1", "1", ...) to a
    # single torch device string via device_utils — DEVICE from .env is the
    # source of truth for the whole pipeline.
    args.bert_device = str(get_device_config(args.bert_device).primary_device)
    args.clip_device = str(get_device_config(args.clip_device).primary_device)

    explanations_path = Path(args.json_path)
    concept_path = Path(args.concept_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not explanations_path.exists():
        raise FileNotFoundError(
            f"Missing explanations json: {explanations_path}"
        )
    if not concept_path.exists():
        raise FileNotFoundError(f"Missing concept file: {concept_path}")

    with explanations_path.open("r", encoding="utf-8") as f:
        payload = expand(json.load(f))

    results = payload.get("results", payload)
    ks = tuple(range(1, max(1, args.max_k) + 1))

    ignore_words = frozenset(
        _normalize_text(w)
        for w in (args.ignore_words or "").split(",")
        if _normalize_text(w)
    )
    ranked_records: List[Dict[str, Any]] = []
    for result in results:
        ranked_records.append(
            _extract_topk_predictions(result, ks, ignore_words)
        )

    ranked_scores = _per_sample_scores(
        ranked_records, ks, args.bert_device, args.clip_device
    )
    ranked_summary = _summarize(ranked_scores)

    concept_bank = _build_concept_bank(concept_path)
    rng = random.Random(args.seed)
    random_records: List[Dict[str, Any]] = []
    for record in ranked_records:
        random_records.append(
            _build_random_predictions(
                gt_token_word_map=record["gt_token_word_map"],
                concept_bank=concept_bank,
                ks=ks,
                rng=rng,
            )
        )

    random_scores = _per_sample_scores(
        random_records, ks, args.bert_device, args.clip_device
    )
    random_summary = _summarize(random_scores)
    per_sample_path = write_per_sample_scores(
        out_dir / f"{args.output_prefix}_per_sample.csv",
        results,
        ranked_scores,
        random_scores,
    )

    table_rows = []
    for k in ks:
        row = {
            "rank_k": k,
            "bert_mean": ranked_summary[k]["bert_mean"],
            "bert_std": ranked_summary[k]["bert_std"],
            "clip_mean": ranked_summary[k]["clip_mean"],
            "clip_std": ranked_summary[k]["clip_std"],
            "random_bert_mean": random_summary[k]["bert_mean"],
            "random_bert_std": random_summary[k]["bert_std"],
            "random_clip_mean": random_summary[k]["clip_mean"],
            "random_clip_std": random_summary[k]["clip_std"],
        }
        table_rows.append(row)

    json_path = out_dir / f"{args.output_prefix}_table.json"
    csv_path = out_dir / f"{args.output_prefix}_table.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "json_path": str(explanations_path),
                "concept_path": str(concept_path),
                "max_k": args.max_k,
                "seed": args.seed,
                "rows": table_rows,
            },
            f,
            indent=2,
        )

    headers = [
        "rank_k",
        "bert_mean",
        "bert_std",
        "clip_mean",
        "clip_std",
        "random_bert_mean",
        "random_bert_std",
        "random_clip_mean",
        "random_clip_std",
    ]

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in table_rows:
            f.write(
                ",".join(
                    [str(row["rank_k"])]
                    + [f"{row[h]:.6f}" for h in headers[1:]]
                )
                + "\n"
            )

    print("Top-k table:")
    print(
        "k | bert(mean±std) | clip(mean±std) | random_bert(mean±std) | "
        "random_clip(mean±std)"
    )
    for row in table_rows:
        print(
            f"{row['rank_k']} | "
            f"{row['bert_mean']:.4f}±{row['bert_std']:.4f} | "
            f"{row['clip_mean']:.4f}±{row['clip_std']:.4f} | "
            f"{row['random_bert_mean']:.4f}±{row['random_bert_std']:.4f} | "
            f"{row['random_clip_mean']:.4f}±{row['random_clip_std']:.4f}"
        )

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved per-sample CSV: {per_sample_path}")


if __name__ == "__main__":
    main()
