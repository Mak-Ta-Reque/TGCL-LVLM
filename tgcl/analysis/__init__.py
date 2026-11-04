"""Step 5: concepts from step-4 features (decomposition) and their grounding.

analyse_features is called by tgcl.analyse_features, once per tag and method:
decompose_activations learns a concept bank from a tag's features;
redefine_activations re-grounds a combined bank (text and image grounding).
"""

import argparse
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from tgcl.analysis.feature_decomposition import decompose_and_ground_activations
from tgcl.analysis.feature_decomposition import decompose_tag
from tgcl.analysis.feature_decomposition import get_feature_matrix
from tgcl.analysis.feature_decomposition import refine_ground_activations
from tgcl.analysis.utils import get_matched_token_of_interest_mask
from tgcl.analysis.utils import get_token_of_interest_features
from tgcl.analysis.utils import mask_to_positive_answers
from tgcl.models.image_text_model import ImageTextModel

__all__ = ["load_features", "analyse_features", "decompose_feature_file"]

SUPPORTED_ANALYSIS = [
    "decompose_activations",
    "redefine_activations",
]


def load_features(
    features_path: Union[str, List[str]],
    logger: Optional[logging.Logger] = None,
    feature_key: str = "hidden_states",
    args: Optional[argparse.Namespace] = None,
    keep_only_token_of_interest: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load step-4 feature files as (samples x dim) matrices.

    Args:
        features_path: One file or several.
        logger: Optional logger.
        feature_key: Entry holding the hidden states.
        args: Analysis arguments (module_to_decompose,
            decomposition_extract_pos, load_matched_features).
        keep_only_token_of_interest: Drop samples whose answer did not
            contain the concept word (token_of_interest_mask). With
            DECOMPOSE_POSITIVES_ONLY=1 the mask also drops negative answers.

    Returns:
        (features, metadata), both keyed by file name: the feature matrix and
        the file's other entries (predictions, images, masks, ...).
    """

    if isinstance(features_path, str):
        features_path = [features_path]
    features = {}
    meta_data = {}
    token_of_interest_mask = None
    if args.load_matched_features and len(features_path) > 1:
        token_of_interest_mask = get_matched_token_of_interest_mask(
            features_path
        )

    positives_only = os.environ.get("DECOMPOSE_POSITIVES_ONLY", "0") == "1"
    for feat_path in features_path:
        data = torch.load(feat_path, map_location="cpu")
        assert (
            feature_key in data
        ), f"{feature_key} not found, got {data.keys()}."
        if positives_only and data.get("token_of_interest_mask") is not None:
            # features and metadata (grounding images, predictions) both follow
            # this mask
            data["token_of_interest_mask"] = mask_to_positive_answers(
                data["token_of_interest_mask"], data["model_predictions"]
            )

        data_ = data[feature_key]

        feat = get_feature_matrix(
            data_,
            module_name=args.module_to_decompose,
            token_idx=args.decomposition_extract_pos,
            token_of_interest_mask=data.get("token_of_interest_mask", None),
            # token_id = args.token_id,
            # metadata = meta_data
        )
        meta = {k: v for k, v in data.items() if feature_key not in k}
        feat_key = os.path.basename(feat_path)
        if keep_only_token_of_interest:
            if token_of_interest_mask is None:
                feat = get_token_of_interest_features(
                    feat, meta.get("token_of_interest_mask", None)
                )
            else:
                feat = get_token_of_interest_features(
                    feat, token_of_interest_mask
                )
        features[feat_key] = feat
        meta_data[feat_key] = meta
        if logger is not None:
            logger.info(
                f"Loading data from {feat_path}.\n keys: {data.keys()}.\n "
                f"Extracted features of shape {feat.shape}"
            )

    return features, meta_data


def decompose_feature_file(args: argparse.Namespace):
    """The decomposition analyse_features would compute for args.features_path
    (same loading and token-of-interest filtering), or None if no sample is
    left. Pass the result back as analyse_features(..., decomposition=...)."""
    features, _ = load_features(
        features_path=args.features_path,
        args=args,
        keep_only_token_of_interest=True,
    )
    mats = [
        v
        for v in features.values()
        if not (hasattr(v, "shape") and v.shape[0] == 0)
    ]
    if not mats:
        return None
    return decompose_tag(mats[0], args)


def load_concept(
    analysis_path: str,
    logger: Optional[logging.Logger] = None,
    analysis_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a saved concept bank.

    Args:
        analysis_path: The bank's .pth file.
        logger: Optional logger.
        analysis_keys: Entries that must exist (default: concepts,
            activations, method, text and image grounding, analysis model).
        args: Unused; kept for the analysis call convention.

    Returns:
        The required entries plus any optional ones present (grounding
        boxes/masks/predictions, selected layer, purity statistics).
    """
    if analysis_keys is None:
        analysis_keys = [
            "concepts",
            "activations",
            "decomposition_method",
            "text_grounding",
            "image_grounding_paths",
            "analysis_model",
        ]
    data = torch.load(analysis_path, map_location="cpu")
    analysis_data_ = {}
    for analysis_key in analysis_keys:
        assert (
            analysis_key in data
        ), f"{analysis_key} not found, got {data.keys()}."
        analysis_data_[analysis_key] = data[analysis_key]
    # Propagate optional keys that may exist in the .pth file (e.g. masks,
    # bboxes, predictions)
    optional_keys = [
        "image_grounding_bboxes",
        "image_grounding_masks",
        "image_grounding_predictions",
        "selected_layer",
        "direction_positive_ratio",
        "direction_num_assigned",
        "direction_weighted_ratio",
        "clean_example_ratio",
    ]
    for opt_key in optional_keys:
        if opt_key in data and opt_key not in analysis_data_:
            analysis_data_[opt_key] = data[opt_key]
    if logger is not None:
        logger.info(
            f"Loading data from {analysis_path}.\n Data size: "
            f"{len(analysis_data_['concepts'])}, keys: "
            f"{analysis_data_.keys()}"
        )

    return analysis_data_


@torch.no_grad()
def analyse_features(
    analysis_name: str = "decompose_activations",
    model_class: Optional[ImageTextModel] = None,
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
    **kwargs: Any,
) -> None:
    """Run one step-5 analysis and save its result.

    Args:
        analysis_name: Contains decompose_activations (learn concepts from
            args.features_path) or redefine_activations (re-ground the bank
            at args.analysis_saving_path).
        model_class: The VLM wrapper (for text grounding via its LM head).
        logger: Optional logger.
        device: Unused by the CPU decompositions; kept for the call convention.
        args: Analysis arguments.
        **kwargs: decomposition: a precomputed decompose_feature_file result.

    The result is saved to <save_dir>/<analysis_name>_<save_filename>.pth;
    nothing is saved when no sample survives the token-of-interest filter.
    """

    # Mandatory whenever a config's features were extracted with the
    # save_hidden_states_for_token_of_interest hook (cgdl, non_contrastive):
    # samples where the tag word was never found (mask=False, hook fell back
    # to position 0) are noise, not signal, and must be dropped rather than
    # pooled in as if they were real matches. This is a no-op for templates
    # using save_hidden_states_mean (null): those files carry no
    # token_of_interest_mask key at all, so get_token_of_interest_features
    # (called with keep_only_token_of_interest=True below) passes the
    # features through unfiltered regardless of this flag's value.
    keep_only_token_of_interest = True
    # Which of these the branches below fill depends on the inputs given.
    features = metadata = concept_dict = None

    if args.features_path is not None:
        features, metadata = load_features(
            features_path=args.features_path,
            logger=logger,
            args=args,
            keep_only_token_of_interest=keep_only_token_of_interest,
        )

    elif args.analysis_saving_path is not None:
        concept_dict = load_concept(
            analysis_path=args.analysis_saving_path,
            logger=logger,
        )

    else:
        raise ValueError(
            "analyse_features needs --features_path (decompose) or "
            "--analysis_saving_path (redefine a saved concept bank)"
        )

    results_dict = {}
    # A tag's raw file can be non-empty but still end up with 0 rows here:
    # keep_only_token_of_interest filtering (above) drops every sample where
    # the tag word was never found, which for a small sample count and a
    # template with a lower match rate (non_contrastive) can legitimately
    # zero out one specific tag even though most tags survive fine.
    # decompose_activations()'s stats print (np.min/np.max) crashes outright
    # on an empty matrix, so skip decomposition for this tag/file instead --
    # results_dict stays {} and the save below is a no-op, same as the
    # existing "empty concept bank" handling elsewhere in this pipeline.
    if "decompose_activations" in analysis_name and isinstance(features, dict):
        empty_keys = [
            k
            for k, v in features.items()
            if hasattr(v, "shape") and v.shape[0] == 0
        ]
        for k in empty_keys:
            if logger is not None:
                logger.warning(
                    f"Skipping decomposition for '{k}': 0 samples survived "
                    "token-of-interest filtering (tag word never found in "
                    "this tag's responses) -- not an error, just no signal "
                    "to decompose for this tag."
                )
            del features[k]
            metadata.pop(k, None)
        if not features:
            return
    if "decompose_activations" in analysis_name:
        results_dict = decompose_and_ground_activations(
            features,
            metadata,
            analysis_name=analysis_name,
            model_class=model_class,
            logger=logger,
            args=args,
            decomposition=kwargs.get("decomposition"),
        )
    elif "redefine_activations" in analysis_name:
        results_dict = refine_ground_activations(
            concept_dict,
            analysis_name=analysis_name,
            model_class=model_class,
            logger=logger,
            args=args,
        )

    else:
        raise NotImplementedError(
            f"Only the following analysis are supported: {SUPPORTED_ANALYSIS}"
        )
    if results_dict:
        file_name = os.path.join(
            args.save_dir, f"{analysis_name}_{args.save_filename}.pth"
        )
        torch.save(results_dict, file_name)
        if logger is not None:
            logger.info(f"Saving analysis results to: {file_name}")
