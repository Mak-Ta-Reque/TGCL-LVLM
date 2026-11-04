#!/usr/bin/env python3
"""Post-hoc sparsification of a learned concept bank.

Every concept c (unit norm) is replaced by a sparse unit vector that uses at
most s = ceil(density * D) feature coordinates:

  omp        the s coordinates and values that best keep the concept's scores
             on real hidden states: min ||H c' - H c||^2 s.t. ||c'||_0 <= s
             (Orthogonal Matching Pursuit; H = the run's training features,
             step 4; columns normalised for the selection)
  magnitude  baseline: the s largest |c| coordinates

The rest of the bank (names, grounding images) is kept; --reground recomputes
the concepts' words through the model's lm_head (as pipeline step 5), since the
words of a sparse vector differ. Evaluate the result with
scripts/rerank_explanations.py.

    python scripts/sparsify_concepts.py \\
        --run outputs/paper/tgcl_sam_coco_seed1 --method snmf \\
        --density 0.1 --algorithm omp --out sparse_bank.pth
"""

import argparse
import copy
import functools
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
import torch

from tgcl.analysis import load_features
from tgcl.analysis.multimodal_grounding import concept_text_grounding
from tgcl.analysis.multimodal_grounding import needs_final_norm
from tgcl.helpers.logit_lens import resolve_final_norm_module
from tgcl.helpers.run_layout import find_concept_bank
from tgcl.models.revisions import revision


def training_states(
    features_dir: Path, layer: str = "model.language_model.norm"
) -> np.ndarray:
    """(N, D) token-of-interest hidden states of every tag file, as step 5
    decomposes them."""
    args = argparse.Namespace(
        module_to_decompose=layer,
        decomposition_extract_pos=None,
        load_matched_features=False,
    )
    mats = []
    for path in sorted(features_dir.glob("*.pth")):
        if path.name == "combined_features.pth":
            continue
        feats, _ = load_features(
            features_path=[str(path)],
            args=args,
            keep_only_token_of_interest=True,
        )
        mats += [
            np.asarray(v.float().numpy(), dtype=np.float64)
            for v in feats.values()
            if v.shape[0]
        ]
    return np.concatenate(mats, axis=0)


def unit(c: np.ndarray) -> np.ndarray:
    return c / np.maximum(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12)


def center_concepts(concepts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Unit concepts minus alpha times the bank mean: removes (alpha=1) or
    shrinks the component every concept shares (the explainer's matching
    subtracts it too), the main source of their overlap. That component also
    carries part of the prediction, so alpha trades overlap for faithfulness.
    """
    return unit(concepts - alpha * concepts.mean(axis=0, keepdims=True))


def lowdin_orthogonalize(concepts: np.ndarray) -> np.ndarray:
    """Symmetric alignment of the bank: Lowdin orthogonalization
    (C C^T)^(-1/2) C, the orthonormal set closest to the unit concepts,
    independent of their order (TGCL's combination, CONCEPT_ORTHO=lowdin)."""
    c = unit(concepts)
    w, v = np.linalg.eigh(c @ c.T)
    return unit((v @ np.diag(np.maximum(w, 1e-12) ** -0.5) @ v.T) @ c)


def magnitude_sparsify(concepts: np.ndarray, n_keep: int) -> np.ndarray:
    """Noise removal: every concept keeps its n_keep largest-magnitude
    coordinates, the rest are set to 0 (TGCL keeps 50 %,
    CONCEPT_SPARSIFY=magnitude:0.5); rows re-normalized."""
    out = np.zeros_like(concepts)
    for k, c in enumerate(concepts):
        idx = np.argpartition(np.abs(c), -n_keep)[-n_keep:]
        out[k, idx] = c[idx]
    return unit(out)


def omp_sparsify(
    concepts: np.ndarray, states: np.ndarray, n_keep: int
) -> np.ndarray:
    """argmin ||H c' - H c|| with ||c'||_0 <= n_keep, per concept (OMP on the
    Gram matrix)."""
    norms = np.maximum(np.linalg.norm(states, axis=0), 1e-12)
    xn = states / norms  # normalised columns: selection by correlation
    gram = xn.T @ xn
    targets = states @ concepts.T  # (N, K): each concept's scores on the data
    coefs = orthogonal_mp_gram(
        gram,
        xn.T @ targets,
        n_nonzero_coefs=n_keep,
        norms_squared=(targets**2).sum(0),
    )
    coefs = (
        np.asarray(coefs).reshape(len(norms), -1).T / norms
    )  # back to the original coordinates
    return unit(coefs)


def fidelity(
    concepts: np.ndarray, sparse: np.ndarray, states: np.ndarray
) -> Dict[str, float]:
    """How well the sparse bank keeps the scores and the concept ranking on the
    data."""
    s0, s1 = states @ unit(concepts).T, states @ sparse.T
    corr = [
        np.corrcoef(s0[:, k], s1[:, k])[0, 1] for k in range(s0.shape[1])
    ]  # scale-free per concept
    return {
        "score_corr": float(np.mean(corr)),
        "rank1_kept": float(np.mean(s0.argmax(1) == s1.argmax(1))),
        "cosine_to_dense": float(np.mean((unit(concepts) * sparse).sum(1))),
    }


def reground(bank: Dict[str, Any], model_name: str, layer: str) -> None:
    """Recompute the bank's text grounding (its words) through the model's LM
    head and final norm, in place."""
    head, norm, tok = output_head(model_name)
    concepts = torch.as_tensor(bank["concepts"]).float()
    bank["text_grounding"] = concept_text_grounding(
        concepts,
        lm_head=head,
        tokenizer=tok,
        num_top_tokens=5,
        pre_num_top_tokens=20,
        final_norm=norm,
        norm_mask=[needs_final_norm(layer)] * len(concepts),
    )


@functools.lru_cache(maxsize=1)
def output_head(model_name: str):
    """(LM head, final norm, tokenizer) of a VLM, loaded once per process."""
    from transformers import AutoModelForImageTextToText
    from transformers import AutoProcessor

    # Any VLM of the pipeline (Gemma-3n, Qwen2.5-VL, ...): only its output head
    # and final norm are used.
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=os.getenv("HF_TOKEN"),
        revision=revision(model_name),
    )
    lm_head = model.get_output_embeddings()
    head = torch.nn.Linear(
        lm_head.weight.shape[1], lm_head.weight.shape[0], bias=False
    )
    head.weight.data = lm_head.weight.data.float()
    norm = resolve_final_norm_module(model.model.language_model)
    norm = copy.deepcopy(norm).float() if norm is not None else None
    tok = AutoProcessor.from_pretrained(
        model_name, token=os.getenv("HF_TOKEN"), revision=revision(model_name)
    ).tokenizer
    del model
    return head, norm, tok


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--method", default="snmf")
    ap.add_argument(
        "--bank",
        type=Path,
        default=None,
        help="input bank (default: the run's combined bank)",
    )
    ap.add_argument(
        "--density",
        type=float,
        required=True,
        help="fraction of coordinates kept per concept",
    )
    ap.add_argument("--algorithm", choices=["omp", "magnitude"], default="omp")
    ap.add_argument(
        "--center",
        type=float,
        default=0.0,
        help="subtract this fraction of the bank mean (1 = all; then "
        "renormalize) before sparsifying",
    )
    ap.add_argument(
        "--orthogonalize",
        choices=["", "lowdin"],
        default="",
        help="after centring, before sparsifying",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--reground",
        action="store_true",
        help="recompute the concepts' words via the lm_head",
    )
    ap.add_argument("--model_name", default="google/gemma-3n-E4B-it")
    ap.add_argument("--layer", default="model.language_model.norm")
    args = ap.parse_args()

    bank = torch.load(
        args.bank or find_concept_bank(args.run, args.method),
        map_location="cpu",
        weights_only=False,
    )
    concepts = np.asarray(
        torch.as_tensor(bank["concepts"]).float().numpy(), dtype=np.float64
    )
    if args.center:
        concepts = center_concepts(concepts, args.center)
    if args.orthogonalize == "lowdin":
        concepts = lowdin_orthogonalize(concepts)
    n_keep = max(1, math.ceil(args.density * concepts.shape[1]))
    if args.algorithm == "omp" or (args.run / "features").exists():
        states = training_states(args.run / "features", args.layer)
    else:  # truncation needs no data; fidelity is then not measured
        states = None
    sparse = (
        omp_sparsify(concepts, states, n_keep)
        if args.algorithm == "omp"
        else magnitude_sparsify(concepts, n_keep)
    )
    stats = fidelity(concepts, sparse, states) if states is not None else {}
    bank["concepts"] = torch.tensor(sparse, dtype=torch.float32)
    bank["sparsification"] = {
        "algorithm": args.algorithm,
        "density": args.density,
        "center": args.center,
        "orthogonalize": args.orthogonalize,
        "n_keep": n_keep,
        "n_states": int(states.shape[0]) if states is not None else 0,
        **stats,
    }
    if args.reground:
        reground(bank, args.model_name, args.layer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, args.out)
    print(
        f"{args.algorithm} density {args.density} ({n_keep} coords): "
        + (
            ", ".join(f"{k} {v:.4f}" for k, v in stats.items())
            or "no training states"
        )
    )


if __name__ == "__main__":
    main()
