"""Sample selection for the decomposition: token-of-interest, answer masks."""

from typing import Any, List, Optional

import torch

from tgcl.combine_concepts import is_negative_prediction

__all__ = [
    "cosine_similarity",
    "get_token_of_interest_features",
    "get_matched_token_of_interest_mask",
    "mask_to_positive_answers",
]


def mask_to_positive_answers(
    token_of_interest_mask: List[torch.Tensor], predictions: List[Any]
) -> List[torch.Tensor]:
    """The token-of-interest mask restricted to answers that name the class.

    With forced-answer features (FORCED_RESPONSE) every segment's state comes
    from the same text, so "No X" segments are no longer separable by wording;
    decomposing positives and negatives together leaves no direction with the
    >= CLEAN_EXAMPLE_RATIO share of positives (empty banks).
    DECOMPOSE_POSITIVES_ONLY=1 learns the concepts from the positive segments.

    Args:
        token_of_interest_mask: Step-4 mask, one bool tensor per batch.
        predictions: The answer of every sample, in order.

    Returns:
        The mask AND "not combine_concepts.is_negative_prediction", in the same
        per-batch structure.

    Raises:
        ValueError: The mask and the predictions differ in length.
    """
    batches = [
        torch.as_tensor(m).reshape(-1).bool() for m in token_of_interest_mask
    ]
    flat = torch.cat(batches)
    if len(flat) != len(predictions):
        raise ValueError(
            f"token_of_interest_mask has {len(flat)} samples, predictions "
            f"{len(predictions)}"
        )
    positive = torch.tensor(
        [not is_negative_prediction(p) for p in predictions], dtype=torch.bool
    )
    return list(torch.split(flat & positive, [len(b) for b in batches]))


def cosine_similarity(
    batch1: torch.Tensor, batch2: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Pairwise cosine similarity of the rows of two matrices, (B1 x B2)."""
    batch1, batch2 = batch1.float(), batch2.float()
    dot_product = batch1 @ batch2.T  # Shape: (B, B)

    norm_batch1 = torch.norm(batch1, p=2, dim=-1, keepdim=True)
    norm_batch2 = torch.norm(batch2, p=2, dim=-1, keepdim=True)

    cosine_sim = dot_product / (norm_batch1 * norm_batch2.T + eps)

    return cosine_sim


def get_token_of_interest_features(
    features: torch.Tensor,
    token_of_interest_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The rows of features whose answer contained the concept word.

    Args:
        features: (samples x dim) features.
        token_of_interest_mask: One bool per sample, or a list of per-batch
            bool tensors (concatenated, never reduced within a batch, so it
            stays aligned 1:1 with the samples); None keeps every row.

    Returns:
        The kept rows.
    """
    if token_of_interest_mask is not None:
        if isinstance(token_of_interest_mask, list):
            token_of_interest_mask = torch.cat(token_of_interest_mask, dim=0)
        features = features[token_of_interest_mask].reshape(
            -1, features.shape[1]
        )

    return features


def get_matched_token_of_interest_mask(
    features_path: List[str],
) -> torch.Tensor:
    """Samples whose answer contained the concept word in every given file.

    Args:
        features_path: Feature files of the same samples (e.g. two models).

    Returns:
        The AND of their token-of-interest masks, or None if a file has none.
    """
    masks = []
    token_of_interest_mask = None
    for feat_path in features_path:
        data = torch.load(feat_path, map_location="cpu")
        token_of_interest_mask = data.get("token_of_interest_mask", None)
        if token_of_interest_mask is not None:
            masks.append(torch.cat(token_of_interest_mask, dim=0))
        else:
            # All features should have token_of_interest_mask for this function
            # to be valid
            return None
    masks = torch.stack(masks, dim=0)
    token_of_interest_mask = torch.all(masks, dim=0)

    return token_of_interest_mask
