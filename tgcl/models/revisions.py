"""Pinned Hugging Face Hub commits of every model the pipeline loads.

A model repository can change after publication (new weights, a new chat
template), so each model is loaded at the commit the paper runs used. Loaders
that take a revision get it from revision(); libraries that only take a path
(LangSAM's Grounding DINO, BERTScore's RoBERTa) get pinned_snapshot(). CLIP and
SAM 2 come from fixed checkpoint URLs (CLIP checks their SHA-256).
"""

import os
from typing import Optional

from huggingface_hub import snapshot_download

MODEL_REVISIONS = {
    "google/gemma-3n-E4B-it": "c1221e9c62e34a43ab7ffacd1be0ea71f126ef10",
    "Qwen/Qwen2-VL-7B-Instruct": "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    # Grounding DINO inside LangSAM (step 2 text-prompted segments).
    "IDEA-Research/grounding-dino-base": (
        "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    ),
    # BERTScore's default English model (retrieval and alignment scores).
    "roberta-large": "722cf37b1afa9454edce342e7895e588b6ff1d59",
    # Phrase generation of the concept-word alignment score.
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    ),
}
GROUNDING_DINO = "IDEA-Research/grounding-dino-base"
BERTSCORE_MODEL = "roberta-large"
BERTSCORE_LAYERS = 17  # bert_score's default layer for roberta-large


def revision(model_name: str) -> Optional[str]:
    """The pinned commit of a Hub model; None for local paths and other
    models."""
    return MODEL_REVISIONS.get(model_name)


def pinned_snapshot(model_name: str) -> str:
    """Local folder of the model at its pinned commit (the latest commit for
    models without one), downloaded on first use."""
    return snapshot_download(
        model_name,
        revision=revision(model_name),
        token=os.environ.get("HF_TOKEN"),
    )
