"""Text and image grounding of concept vectors.

Text grounding reads a concept through the LM head (top tokens, filtered to
English non-stop-words). Image grounding keeps each concept's most activating
samples with their metadata, and the purity statistics that combine_concepts
uses to keep or drop the concept.
"""

import argparse
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from tgcl.helpers.nltk_data import english_words
import tgcl.helpers.utils as helpers_utils
from tgcl.metrics.utils import get_stopwords
from tgcl.metrics.utils import GIST_FILE_PATH
from tgcl.metrics.utils import valid_word
from tgcl.models.image_text_model import ImageTextModel

__all__ = [
    "get_multimodal_grounding",
    "concept_text_grounding",
    "concept_image_grounding",
]


def needs_final_norm(layer_name) -> bool:
    """True when features from *layer_name* must go through the final RMSNorm
    before lm_head (standard logit lens). Features already taken at the final
    norm (or at lm_head itself) must NOT be normalized a second time. None
    (legacy banks without layer info) is treated as final-norm output."""
    if not isinstance(layer_name, str) or not layer_name:
        return False
    lowered = layer_name.lower()
    if "lm_head" in lowered:
        return False
    return "norm" not in lowered.rsplit(".", 1)[-1]


def remove_substrings(tokens):
    """Unique tokens that are not part of another token, in their given order (a
    set would order them by the per-process string hash)."""
    tokens = list(tokens)
    return [
        token
        for token in dict.fromkeys(tokens)
        if not any(
            token != other and len(token) > 0 and token in other
            for other in tokens
        )
    ]


@torch.no_grad()
def concept_text_grounding(
    concepts: torch.Tensor,
    lm_head: Optional[torch.nn.Module] = None,
    tokenizer: Any = None,
    num_top_tokens: int = 15,
    gist_file_path: str = GIST_FILE_PATH,
    pre_num_top_tokens: int = 50,
    keep_unique_words: bool = True,
    final_norm: Optional[torch.nn.Module] = None,
    norm_mask: Optional[List[bool]] = None,
) -> List[List[str]]:
    """Words of each concept: the top LM-head tokens of its vector (logit lens).

    Args:
        concepts: (K x dim) concept vectors.
        lm_head: The model's LM head.
        tokenizer: Decodes the token ids.
        num_top_tokens: Words kept per concept.
        gist_file_path: Stop-word list.
        pre_num_top_tokens: Tokens considered before filtering (> kept).
        keep_unique_words: Drop repeated words (keeping the logit order).
        final_norm: The model's final norm, applied to the rows of norm_mask
            (mid-layer features) before the LM head.
        norm_mask: Per concept, whether to apply final_norm.

    Returns:
        Per concept, up to num_top_tokens English non-stop-words, best first.
    """

    eng_corpus = english_words()
    stopwords = get_stopwords(gist_file_path=gist_file_path)
    num_concepts = concepts.shape[0]

    concepts = concepts.to(next(lm_head.parameters()).device)

    # Logit lens: mid-layer features must pass the model's final RMSNorm
    # before lm_head, otherwise the logits are distorted and the top tokens
    # are garbage that the word filter removes entirely. Features taken at
    # the final norm itself must NOT be normalized twice — norm_mask says
    # per concept row whether to apply it.
    if final_norm is not None and norm_mask is not None and any(norm_mask):
        try:
            norm_device = next(final_norm.parameters()).device
        except StopIteration:
            norm_device = concepts.device
        rows = torch.tensor(norm_mask, dtype=torch.bool, device=concepts.device)
        normed = final_norm(concepts[rows].to(norm_device)).to(
            device=concepts.device, dtype=concepts.dtype
        )
        concepts = concepts.clone()
        concepts[rows] = normed

    token_logits = lm_head(concepts)
    assert pre_num_top_tokens > num_top_tokens, (
        f"pre_num_top_tokens {pre_num_top_tokens} <= num_top_tokens "
        f"{num_top_tokens}"
    )
    top_token_idx = token_logits.argsort(dim=-1, descending=True)[
        :, :pre_num_top_tokens
    ]
    grounded_words_list = []
    for k in range(num_concepts):
        # Decode each token id separately: batch_decode on a 1-D id tensor
        # concatenates them into ONE string (single sequence), which the word
        # filter then rejects wholesale.
        comp_words = tokenizer.batch_decode(
            [[int(t)] for t in top_token_idx[k].tolist()],
            skip_special_tokens=True,
        )
        comp_words = [
            word.lower().strip()
            for word in comp_words
            if valid_word(word, eng_corpus=eng_corpus, stopwords=stopwords)
        ]
        comp_words = remove_substrings(
            comp_words
        )  # Remove substrings from the list of words
        # Filter out words that are not valid

        # comp_words = [word for word in comp_words if word not in stopwords]

        icomp_words = []
        for word in comp_words:
            clean_word = re.sub(
                r"[^a-zA-Z0-9]", "", word.strip()
            )  # Remove non-alphanumeric characters
            if clean_word and clean_word not in stopwords:
                icomp_words.append(clean_word)

        if keep_unique_words:
            # Keep the logit order: the cut below must keep the top tokens,
            # not whichever ones a set happens to list first.
            icomp_words = list(dict.fromkeys(icomp_words))
        grounded_words_list.append(icomp_words[:num_top_tokens])
    return grounded_words_list


# Sentence-ending punctuation/quotes VLMs commonly wrap short yes/no answers
# in (e.g. "No.", "no!", '"no"') — stripped before matching so yn-mode
# predictions aren't silently miscounted as positive.
_NEGATIVE_STRIP_CHARS = ".,!?;:'\")( \t\n\r"


def _is_negative_prediction(s) -> bool:
    """Mirror of combine_concepts.is_negative_prediction (kept local to avoid
    importing the heavier module here)."""
    low = str(s).lower().strip()
    low = low.strip(_NEGATIVE_STRIP_CHARS)
    return (
        low == "no"
        or low.startswith("no_")
        or low.startswith("not_")
        or low.startswith("unk")
        or low.startswith("thing")
        or low.startswith("nc")
        or low.startswith("no ")
    )


def compute_direction_positive_ratios(
    activations: torch.Tensor,
    predictions: Any,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Any], List[int], List[Any]]:
    """Per-direction positive ratios for the positive/negative concept split.

    Two complementary ratios per direction j:

    1. **Assigned ratio** (hard): each sample goes to the direction where its
       activation is highest (argmax over K); samples whose activations are
       all zero are *excluded* (the factorization explains them with no
       direction, so they are evidence for nothing).
       ``ratio_j = #positive among assigned / #assigned``.
    2. **Weighted ratio** (soft, margin-aware): uses the full loading mass
       instead of a hard assignment —
       ``weighted_j = sum(A[i, j] for positive i) / sum(A[i, j] for all i)``.
       A borderline 0.51/0.49 sample contributes ~half its mass to each
       direction instead of counting fully for one; ties are irrelevant.

    Both stay meaningful when negatives dominate the tag's samples: a direction
    that captures the few true positives scores high, background directions
    score low.

    Returns (ratios, counts, weighted_ratios); a ratio is None when there is no
    supporting mass/samples or when predictions cannot be aligned to the
    activation rows.
    """
    num_samples, num_directions = activations.shape

    # model_predictions may be stored per batch (list of lists) — flatten.
    flat_preds: List[Any] = []
    for entry in predictions or []:
        if isinstance(entry, list):
            flat_preds.extend(entry)
        else:
            flat_preds.append(entry)

    if len(flat_preds) != num_samples:
        if logger is not None:
            logger.warning(
                f"direction_positive_ratio: cannot align {len(flat_preds)} "
                f"predictions to {num_samples} samples; storing None ratios."
            )
        return (
            [None] * num_directions,
            [0] * num_directions,
            [None] * num_directions,
        )

    acts = activations.float()
    positive_mask = torch.tensor(
        [not _is_negative_prediction(p) for p in flat_preds], dtype=torch.bool
    )
    # Samples with zero loading everywhere are unexplained — exclude them
    # from the hard assignment instead of argmax-defaulting them to dir 0.
    explained = acts.max(dim=1).values > 0
    assignment = torch.argmax(acts, dim=1)

    ratios: List[Any] = []
    counts: List[int] = []
    weighted_ratios: List[Any] = []
    for j in range(num_directions):
        owned = explained & (assignment == j)
        n_owned = int(owned.sum())
        counts.append(n_owned)
        if n_owned == 0:
            ratios.append(None)
        else:
            positives = int((owned & positive_mask).sum())
            ratios.append(positives / n_owned)

        mass = acts[:, j].clamp(min=0)
        total_mass = float(mass.sum())
        if total_mass <= 0:
            weighted_ratios.append(None)
        else:
            weighted_ratios.append(
                float(mass[positive_mask].sum()) / total_mass
            )
    return ratios, counts, weighted_ratios


def concept_image_grounding(
    activations: torch.Tensor,
    num_images_per_concept: int = 5,
) -> List[torch.Tensor]:
    """Indices of the samples that belong to each concept's own direction.

    Each sample is hard-assigned to exactly one concept: whichever direction it
    activates highest on (argmax over columns) — the same assignment rule used
    for direction_positive_ratio and in cluster_analysis.py. Without this,
    argsort-per-column (the previous implementation) puts every sample in every
    concept's list, just reordered, so DECOMP_COMPONENTS concepts never actually
    separate into DECOMP_COMPONENTS disjoint groups. Samples with zero
    activation on every direction are unexplained by the factorization and
    excluded from all concepts.

    num_images_per_concept <= 0 (e.g. NUM_MOST_ACTIVATING_SAMPLES=-1) keeps ALL
    of a concept's assigned samples (still ordered by activation, strongest
    first) instead of truncating to a top-k.
    """
    assignment = torch.argmax(activations, dim=1)
    explained = activations.max(dim=1).values > 0
    num_directions = activations.shape[1]

    per_concept_indices = []
    for j in range(num_directions):
        owned = torch.nonzero(explained & (assignment == j), as_tuple=True)[0]
        owned = owned[torch.argsort(activations[owned, j], descending=True)]
        if num_images_per_concept is not None and num_images_per_concept > 0:
            owned = owned[:num_images_per_concept]
        per_concept_indices.append(owned)
    # list of length num_concepts, each a 1-D index tensor
    return per_concept_indices


def get_multimodal_grounding(
    concepts: torch.Tensor,
    activations: torch.Tensor,
    model_class: ImageTextModel,
    text_grounding: bool = True,
    image_grounding: bool = True,
    module_to_decompose: str = "",
    save_dir: str = "",
    save_name: str = "",
    num_grounded_text_tokens: int = 10,
    num_most_activating_samples: int = 5,
    metadata: Optional[Dict[str, Any]] = None,
    save_analysis: bool = False,
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
) -> Optional[Dict[str, Any]]:
    """Ground a freshly learned concept bank in words and in samples.

    Args:
        concepts: (K x dim) concept vectors.
        activations: (samples x K) concept activations.
        model_class: {"lm_head", "tokenizer", optional "final_norm"}.
        text_grounding: Add each concept's top LM-head words.
        image_grounding: Add each concept's most activating samples (paths,
            boxes, masks, predictions) and its purity statistics
            (direction_positive_ratio, ...), used by combine_concepts.
        module_to_decompose: Layer the features come from (decides whether the
            final norm is applied before the LM head).
        save_dir: Folder for save_analysis.
        save_name: File-name suffix for save_analysis.
        num_grounded_text_tokens: Words kept per concept.
        num_most_activating_samples: Samples kept per concept.
        metadata: Per-sample entries of the feature file.
        save_analysis: Also save the result under <save_dir>/analysis/.
        logger: Optional logger.
        args: Analysis arguments (decomposition_method, pre_num_top_tokens).

    Returns:
        The concepts, activations, groundings and statistics; None when the
        features are the LM head itself (nothing to ground).
    """
    if metadata is None:
        metadata = {}
    lm_head = model_class["lm_head"]
    tokenizer = model_class["tokenizer"]
    grounding_dict = {}

    activations = torch.Tensor(activations)
    concepts = torch.Tensor(concepts)
    grounding_dict["concepts"] = concepts
    grounding_dict["activations"] = activations
    grounding_dict["decomposition_method"] = args.decomposition_method

    grounded_words = []

    if "lm_head" in module_to_decompose:
        top_tokens = concepts.argmax(axis=1)
        top_words = tokenizer.decode(top_tokens)
        if logger is not None:
            logger.info(
                f"Lm head top_tokens: {top_tokens}, top words: {top_words}"
            )
            logger.info("Lm head only for analysis. Function returning")
        return

    if text_grounding:
        apply_norm = needs_final_norm(module_to_decompose)
        if (
            apply_norm
            and model_class.get("final_norm", None) is None
            and logger is not None
        ):
            logger.warning(
                f"Features come from mid layer '{module_to_decompose}' "
                "but no final_norm "
                "module is available — logit-lens text grounding may "
                "return empty lists."
            )
        grounded_words = concept_text_grounding(
            concepts,
            lm_head=lm_head,
            tokenizer=tokenizer,
            num_top_tokens=num_grounded_text_tokens,
            pre_num_top_tokens=args.pre_num_top_tokens,
            final_norm=model_class.get("final_norm", None),
            norm_mask=[apply_norm] * int(concepts.shape[0]),
        )
        if logger is not None:
            for i in range(len(grounded_words)):
                logger.info(f"Concept {i} grounded words: {grounded_words[i]}")
        grounding_dict["text_grounding"] = grounded_words

    if image_grounding:
        logger.info(f"Activations size: {activations.shape}")

        image_paths = metadata.get("image", [])
        concept_names = metadata.get("concept", [])
        concept_bbox = metadata.get("bbox", [])
        concept_seg_mask_rle = metadata.get("seg_mask_rle", [])
        predictions = metadata.get("model_predictions", [])
        concept_is_concept = metadata.get("is_concept", [])
        logger.info(f"Image paths length: {len(image_paths)}")

        # `activations` (and therefore `concept_indices`/`compute_direction_
        # positive_ratios` below) was already reduced to token-of-interest
        # samples upstream in load_features/get_token_of_interest_features.
        # Every per-sample metadata list must be filtered the same way BEFORE
        # it's used against `activations`, or indices/zips silently
        # misalign against a list that's still at full, unfiltered length
        # (this previously desynced `predictions` from `activations` in
        # compute_direction_positive_ratios, forcing every direction's
        # positive_ratio to None/assigned=0 for templates like
        # non_contrastive whose token_of_interest_mask actually drops rows).
        token_of_interest_mask = metadata.get("token_of_interest_mask", None)
        # Only mask metadata down when the features it describes were
        # actually filtered upstream (e.g. get_token_of_interest_features
        # with keep_only_token_of_interest=True). When features were kept
        # unfiltered (non_contrastive: load_features is called with
        # keep_only_token_of_interest=False, since there's no per-tag
        # positive/negative split to check a token of interest against),
        # `image_paths` (and the other metadata lists) are already at full,
        # unfiltered length -- exactly matching `activations.shape[0]` -- so
        # masking here would wrongly shrink them out of alignment instead of
        # into it.
        if (
            token_of_interest_mask is not None
            and len(image_paths) != activations.shape[0]
        ):
            # Each list entry is one batch's per-sample found/not-found mask
            # (shape [batch_size]) -- concatenate to flatten to one bool per
            # sample, the same convention get_token_of_interest_features in
            # src/analysis/utils.py uses for `activations`/`features`. Do NOT
            # .any()-collapse per batch: that produces one bool per BATCH
            # instead of per sample, truncating these metadata lists to the
            # batch count and desyncing them from concept_indices (computed
            # against the correctly-sample-aligned activations).
            keep_mask = torch.cat(token_of_interest_mask, dim=0).bool().tolist()

            def _apply_mask(values):
                return [
                    values[i]
                    for i in range(min(len(values), len(keep_mask)))
                    if keep_mask[i]
                ]

            (
                image_paths,
                concept_names,
                concept_bbox,
                concept_seg_mask_rle,
                predictions,
                concept_is_concept,
            ) = (
                _apply_mask(values)
                for values in (
                    image_paths,
                    concept_names,
                    concept_bbox,
                    concept_seg_mask_rle,
                    predictions,
                    concept_is_concept,
                )
            )

        # Per-direction positive ratios (hard assigned + soft weighted) — the
        # selection signals used by combine_concepts against
        # CLEAN_EXAMPLE_RATIO / DIRECTION_RATIO_MODE. Must use the
        # mask-filtered `predictions` above so it's length-aligned with
        # `activations` (both reduced to the same token-of-interest samples).
        direction_ratios, direction_counts, direction_weighted = (
            compute_direction_positive_ratios(
                activations=activations,
                predictions=predictions,
                logger=logger,
            )
        )
        grounding_dict["direction_positive_ratio"] = direction_ratios
        grounding_dict["direction_num_assigned"] = direction_counts
        grounding_dict["direction_weighted_ratio"] = direction_weighted
        if logger is not None:
            for j, (ratio, count, weighted) in enumerate(
                zip(direction_ratios, direction_counts, direction_weighted)
            ):
                ratio_str = f"{ratio:.3f}" if ratio is not None else "None"
                weighted_str = (
                    f"{weighted:.3f}" if weighted is not None else "None"
                )
                logger.info(
                    f"Direction {j}: assigned={count}, "
                    f"positive_ratio={ratio_str}, "
                    f"weighted_ratio={weighted_str}"
                )

        image_indices = concept_image_grounding(
            activations=activations,
            num_images_per_concept=num_most_activating_samples,
        )

        # Per concept, the metadata of its most activating samples.
        fields = {
            "image_grounding_paths": image_paths,
            "image_grounding_predictions": predictions,
            "image_grounding_bboxes": concept_bbox,
            "image_grounding_masks": concept_seg_mask_rle,
            "image_grounding_is_concept": concept_is_concept,
            "image_concept_names": concept_names,
        }
        picked = {key: [] for key in fields}
        all_concept_activations = []

        def _safe_pick(values, idx):
            # Metadata lists may be empty or shorter than the sample count
            # (e.g. per-batch instead of per-sample entries); return None
            # rather than raising for out-of-range indices.
            return (
                values[idx]
                if values is not None and idx < len(values)
                else None
            )

        for i, concept_indices in enumerate(image_indices):
            for key, values in fields.items():
                picked[key].append(
                    [_safe_pick(values, index) for index in concept_indices]
                )
            # Activation values aligned index-for-index with concept_indices
            # (not a blind top-k of the raw column), so downstream consumers
            # like combine_concepts can rely on them.
            all_concept_activations.append(activations[concept_indices, i])
            if logger is not None:
                logger.info(
                    "Concept %s image paths: %s",
                    i,
                    picked["image_grounding_paths"][-1],
                )

        grounding_dict.update(picked)
        grounding_dict["image_grounding_activations"] = all_concept_activations
    grounding_dict["concepts"] = concepts
    grounding_dict["activations"] = activations
    # Layer the decomposed hidden states came from (per-tag when logit-lens
    # layer selection is enabled) — kept so every concept stays traceable.
    grounding_dict["selected_layer"] = module_to_decompose

    if save_analysis:

        assert (
            save_name is not None
        ), "save_name should not be None when save_analysis is set to True."
        analysis_saving_name = (
            f"decompose_activations_{args.decomposition_method}_{save_name}"
        )
        analysis_saving_path = os.path.join(
            save_dir, "analysis", f"{analysis_saving_name}"
        )

        if logger is not None:
            logger.info(
                "Saving decomposition results dictionary to: "
                f"{analysis_saving_path}"
            )

        helpers_utils.save_analysis_to_file(
            grounding_dict,
            analysis_saving_path,
            grounding_dict.keys(),
            logger=logger,
        )

    return grounding_dict


def get_refined_multimodal_grounding(
    concepts: torch.Tensor,
    activations: torch.Tensor,
    model_class: ImageTextModel,
    text_grounding: bool = True,
    module_to_decompose: str = "",
    num_grounded_text_tokens: int = 10,
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
    selected_layers: Optional[List[Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Recompute the text grounding of a combined concept bank.

    Args:
        concepts: (K x dim) concept vectors.
        activations: The bank's activations (passed through).
        model_class: {"lm_head", "tokenizer", optional "final_norm"}.
        text_grounding: Add each concept's top LM-head words.
        module_to_decompose: Layer of the features, used when selected_layers
            gives no per-concept layer.
        num_grounded_text_tokens: Words kept per concept.
        logger: Optional logger.
        args: Analysis arguments (decomposition_method, pre_num_top_tokens).
        selected_layers: Per concept, the layer its features came from (a
            combined bank can mix layers); decides the final norm per row.

    Returns:
        Concepts, activations, method and text grounding; None when the
        features are the LM head itself.
    """
    lm_head = model_class["lm_head"]
    tokenizer = model_class["tokenizer"]
    grounding_dict = {}

    concepts = torch.Tensor(concepts)
    grounding_dict["concepts"] = concepts
    grounding_dict["activations"] = activations
    grounding_dict["decomposition_method"] = args.decomposition_method

    grounded_words = []

    if "lm_head" in module_to_decompose:
        top_tokens = concepts.argmax(axis=1)
        top_words = tokenizer.decode(top_tokens)
        if logger is not None:
            logger.info(
                f"Lm head top_tokens: {top_tokens}, top words: {top_words}"
            )
            logger.info("Lm head only for analysis. Function returning")
        return

    if text_grounding:
        # The combined bank can mix per-tag layers: gate the final-norm
        # application per concept row via its recorded selected_layer.
        num_rows = int(concepts.shape[0])
        if (
            isinstance(selected_layers, list)
            and len(selected_layers) == num_rows
        ):
            norm_mask = [needs_final_norm(layer) for layer in selected_layers]
        else:
            norm_mask = [needs_final_norm(module_to_decompose)] * num_rows
        if (
            any(norm_mask)
            and model_class.get("final_norm", None) is None
            and logger is not None
        ):
            logger.warning(
                "Some concepts come from mid layers but no final_norm "
                "module is "
                "available — logit-lens text grounding may return empty lists."
            )
        grounded_words = concept_text_grounding(
            concepts,
            lm_head=lm_head,
            tokenizer=tokenizer,
            num_top_tokens=num_grounded_text_tokens,
            pre_num_top_tokens=args.pre_num_top_tokens,
            final_norm=model_class.get("final_norm", None),
            norm_mask=norm_mask,
        )
        if logger is not None:
            for i in range(len(grounded_words)):
                logger.info(f"Concept {i} grounded words: {grounded_words[i]}")
        grounding_dict["text_grounding"] = grounded_words

    grounding_dict["concepts"] = concepts
    grounding_dict["activations"] = activations
    grounding_dict["selected_layer"] = module_to_decompose

    return grounding_dict
