"""Concept dictionaries from hidden states: decomposition and grounding.

decompose_activations learns K concept vectors from a (samples x dim) feature
matrix with one of the paper's methods (snmf: semi-NMF, simple, sae2, pca);
decompose_and_ground_activations adds each concept's text grounding (top LM-head
words) and image grounding (its most activating samples).
"""

import argparse
import copy
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import PCA
from sklearn.decomposition import sparse_encode
import torch

from tgcl.analysis.multimodal_grounding import get_multimodal_grounding
from tgcl.analysis.multimodal_grounding import get_refined_multimodal_grounding
from tgcl.device_utils import get_device_config
from tgcl.helpers.sae import SparseAutoencoder
from tgcl.helpers.sae import train_sae
from tgcl.models.image_text_model import ImageTextModel

__all__ = [
    "get_feature_matrix",
    "decompose_activations",
    "decompose_tag",
    "decompose_and_ground_activations",
]


def decompose_tag(mat, args: argparse.Namespace):
    """decompose_activations for one tag's feature matrix, from a fresh seed.

    Reseeding per tag makes every tag's result independent of which tags were
    decomposed before it (randomized-SVD PCA and DictionaryLearning's atom
    resampling draw from the global NumPy RNG), so tags can be decomposed in any
    order or in parallel processes with identical results.
    """
    seed = getattr(args, "seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    num_concepts = (
        args.num_concepts[0]
        if isinstance(args.num_concepts, list)
        else args.num_concepts
    )
    return decompose_activations(
        mat=mat,
        num_concepts=num_concepts,
        decomposition_method=args.decomposition_method,
        args=args,
    )


def decompose_and_ground_activations(
    features: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, Any]] = None,
    analysis_name: str = "decompose_activations",
    model_class: Optional[ImageTextModel] = None,
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
    decomposition=None,
):
    """decomposition: (concepts, activations, model) already computed for these
    features by decompose_tag (e.g. in a worker process); else computed here."""
    if metadata is None:
        metadata = {}
    results_dict = {}
    features = list(features.values())[0]
    metadata = list(metadata.values())[0]
    if decomposition is None:
        decomposition = decompose_tag(features, args)
    concepts, activations, decomposition_model = decomposition
    results_dict["concepts"] = concepts
    results_dict["activations"] = activations
    results_dict["decomposition_method"] = args.decomposition_method
    if logger is not None:
        logger.info(
            f"\nDecomposition type {args.decomposition_method}, "
            f"Components/concepts shape: {concepts.shape}, Activations "
            f"shape: {activations.shape}"
        )
    if "grounding" in analysis_name:
        text_grounding = "text_grounding" in analysis_name
        image_grounding = "image_grounding" in analysis_name
        grounding_dict = get_multimodal_grounding(
            concepts=concepts,
            activations=activations,
            model_class=model_class,
            text_grounding=text_grounding,
            image_grounding=image_grounding,
            module_to_decompose=args.module_to_decompose,
            num_grounded_text_tokens=args.num_grounded_text_tokens,
            num_most_activating_samples=args.num_most_activating_samples,
            metadata=metadata,
            logger=logger,
            args=args,
        )
        grounding_dict["analysis_model"] = decomposition_model
        results_dict.update(grounding_dict)
    return results_dict


def refine_ground_activations(
    concept_dict: Dict[str, torch.Tensor],
    analysis_name: str = "decompose_activations",
    model_class: Optional[ImageTextModel] = None,
    logger: Optional[logging.Logger] = None,
    args: Optional[argparse.Namespace] = None,
):
    """Re-ground a combined concept bank (step 5, after combine_concepts).

    Args:
        concept_dict: The bank (load_concept): concepts, activations, method,
            groundings and purity statistics.
        analysis_name: Contains "grounding" to recompute the groundings
            ("text_grounding" for the LM-head words).
        model_class: The VLM wrapper (its LM head grounds the concepts).
        logger: Optional logger.
        args: Analysis arguments (module_to_decompose, ...).

    Returns:
        The bank with recomputed groundings; image groundings and purity
        statistics are carried over. An empty bank is returned as is.
    """
    results_dict = {}
    concepts = copy.deepcopy(concept_dict["concepts"])
    activations = copy.deepcopy(concept_dict["activations"])
    results_dict["concepts"] = concepts  #  concept_dict ["concepts"]

    results_dict["activations"] = activations
    results_dict["decomposition_method"] = concept_dict["decomposition_method"]
    # Empty concept bank (e.g. no concept passed the CLEAN_EXAMPLE_RATIO
    # purity filter): there is nothing to reground — return the empty dict
    # instead of crashing in lm_head with a (1x0) matmul.
    if (
        not torch.is_tensor(concepts)
        or concepts.numel() == 0
        or concepts.ndim < 2
    ):
        if logger is not None:
            logger.warning(
                "Concept bank is empty — skipping regrounding. No concept "
                "survived combining (check CLEAN_EXAMPLE_RATIO and the "
                "image_grounding_predictions in the intermediate files)."
            )
        results_dict["text_grounding"] = []
        results_dict["selected_layer"] = concept_dict.get("selected_layer", [])
        for opt_key in [
            "image_grounding_paths",
            "image_grounding_bboxes",
            "image_grounding_masks",
            "image_grounding_predictions",
        ]:
            results_dict[opt_key] = concept_dict.get(opt_key, [])
        results_dict["analysis_model"] = concept_dict.get(
            "analysis_model", None
        )
        return results_dict
    if logger is not None:
        logger.info(
            f"\nDecomposition type {args.decomposition_method}, "
            "Components/concepts shape: "
            f"{results_dict['concepts'].shape}, Activations shape: "
            f"{len(concept_dict ['activations'])}"
        )
    if "grounding" in analysis_name:
        text_grounding = "text_grounding" in analysis_name
        grounding_dict = get_refined_multimodal_grounding(
            concepts=concepts,
            activations=activations,
            model_class=model_class,
            text_grounding=text_grounding,
            module_to_decompose=args.module_to_decompose,
            num_grounded_text_tokens=args.num_grounded_text_tokens,
            logger=logger,
            args=args,
            selected_layers=concept_dict.get("selected_layer", None),
        )
        grounding_dict["analysis_model"] = concept_dict["analysis_model"]
        # Prefer the per-concept layer list from the combined bank over the
        # scalar --module_to_decompose (layers differ per tag when logit-lens
        # layer selection is enabled).
        if concept_dict.get("selected_layer", None) is not None:
            grounding_dict["selected_layer"] = concept_dict["selected_layer"]
        for passthrough_key in (
            "direction_positive_ratio",
            "direction_num_assigned",
            "direction_weighted_ratio",
            "clean_example_ratio",
        ):
            if concept_dict.get(passthrough_key, None) is not None:
                grounding_dict[passthrough_key] = concept_dict[passthrough_key]
        grounding_dict["image_grounding_paths"] = concept_dict.get(
            "image_grounding_paths", []
        )
        grounding_dict["image_grounding_bboxes"] = concept_dict.get(
            "image_grounding_bboxes", []
        )
        grounding_dict["image_grounding_masks"] = concept_dict.get(
            "image_grounding_masks", []
        )
        grounding_dict["image_grounding_predictions"] = concept_dict.get(
            "image_grounding_predictions", []
        )
        results_dict.update(grounding_dict)
    return results_dict


def get_feature_matrix(
    features: List[Dict[str, Any]],
    module_name: str,
    token_idx: Optional[int] = None,
    token_of_interest_mask=None,
) -> torch.Tensor:
    """Stack one module's stored hidden states into a (samples x dim) matrix.

    Args:
        features: Per-sample hidden states: {module: [one tensor per sample]}
            (step-4 files), or a list of {module: tensor} dicts.
        module_name: The hooked module whose states are used.
        token_idx: For the list form, the token position to take (else the
            mean over positions).
        token_of_interest_mask: For the list form, per-sample position masks;
            the states at the masked positions are averaged.

    Returns:
        The feature matrix.
    """
    if isinstance(features, dict):
        assert module_name in features.keys(), (
            f"Given module name {module_name} not found among stored modules: "
            f"{features.keys()}"
        )
        features = features[module_name]
        features = torch.stack(features, dim=0).squeeze(1)
        return features

    else:
        assert module_name in features[0].keys(), (
            f"Given module name {module_name} not found among stored modules: "
            f"{features[0].keys()}"
        )

        feat_dim = features[0][module_name].ndim
        if feat_dim == 2:
            # Assuming feat shape [Num_token_positions, Representation_size]
            for feat in features:
                feat[module_name] = feat[module_name].unsqueeze(
                    0
                )  # For unifying all shapes

        if token_of_interest_mask is not None:
            if isinstance(token_of_interest_mask, list):
                if token_of_interest_mask[0].shape[-1] > 1:
                    masked_features = [
                        feat[module_name][:, mask[0]]
                        for feat, mask in zip(features, token_of_interest_mask)
                    ]
                    features = torch.cat(
                        [
                            feat.mean(dim=1, keepdim=False)
                            for feat in masked_features
                        ],
                        dim=0,
                    )
                    return features
        if token_idx is not None:
            matrix = torch.cat(
                [feat[module_name][:, token_idx] for feat in features], dim=0
            )
        else:
            # Take average over token positions
            matrix = torch.cat(
                [
                    feat[module_name].mean(dim=1, keepdim=False)
                    for feat in features
                ],
                dim=0,
            )

        return matrix


def _compute_sae_top_k(args: argparse.Namespace, hidden_dim: int):
    """Translate --sae_target_sparsity (a desired zero-fraction, e.g. 0.99) into
    an absolute top_k count of code entries to keep per sample. Returns None
    when no target is set, which leaves the SAE's soft L1 penalty as the only
    sparsity mechanism (previous behavior)."""
    target_sparsity = getattr(args, "sae_target_sparsity", None)
    if target_sparsity is None:
        return None
    top_k = max(1, round((1 - target_sparsity) * hidden_dim))
    return top_k


def l1_sparse_atoms(
    mat: np.ndarray,
    atoms: np.ndarray,
    codes: np.ndarray,
    code_alpha: float,
    dict_alpha: float,
    n_iter: int = 50,
    inner_steps: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Semi-NMF with an L1 penalty on the concept vectors, refined from a fit.

    Alternates the concepts (proximal gradient on
    0.5 ||X - U C||^2 + beta ||C||_1 with ||c_k||_2 <= 1) and the codes
    (non-negative lasso with code_alpha, as DictionaryLearning). beta is set
    per concept once, as dict_alpha times max |U^T X| of that concept: the
    level at which the concept would vanish, so dict_alpha in [0, 1) means
    the same across models whatever their activation scale. No concept is
    zeroed: an update that would is skipped for that concept.

    Args:
        mat: (N, D) features the decomposition was fit on.
        atoms: (K, D) concepts of the fit (rows of norm <= 1).
        codes: (N, K) non-negative codes of the fit.
        code_alpha: L1 weight on the codes (DL_ALPHA).
        dict_alpha: L1 weight on the concepts, as a fraction of their kill
            level; 0 returns the inputs.
        n_iter: Alternations of concepts and codes.
        inner_steps: Proximal-gradient steps per concept update.

    Returns:
        (atoms, codes): the sparser concepts (K, D) and their codes (N, K).
    """
    if dict_alpha <= 0:
        return atoms, codes
    X = np.asarray(mat, dtype=np.float64)
    C = np.asarray(atoms, dtype=np.float64).copy()
    U = np.asarray(codes, dtype=np.float64)
    beta = dict_alpha * np.abs(U.T @ X).max(axis=1, keepdims=True)
    for _ in range(n_iter):
        gram, ux = U.T @ U, U.T @ X
        step = 1.0 / max(np.linalg.norm(gram, 2), 1e-12)
        for _ in range(inner_steps):
            new = C - step * (gram @ C - ux)
            new = np.sign(new) * np.maximum(np.abs(new) - step * beta, 0.0)
            new /= np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1.0)
            # A concept the step would zero keeps its last value: a dead
            # concept gets zero codes and could never come back.
            alive = np.abs(new).max(axis=1) > 0
            C[alive] = new[alive]
        U = sparse_encode(
            X,
            C,
            algorithm="lasso_cd",
            alpha=code_alpha,
            positive=True,
            max_iter=12000,
        )
    return C.astype(np.float32), U.astype(np.float32)


def decompose_activations(
    mat,
    num_concepts: int,
    decomposition_method: str = "snmf",
    args: Optional[argparse.Namespace] = None,
):
    """
    Input:
        mat: torch tensor or numpy array object of shape (N_samples,
        Representation_dim).
        num_concepts: Number of components/concepts
        decomposition_method: Decomposition/Dictionary learning model type
        (Options: PCA, KMeans, Semi-NMF/Non-negative dict learning, Simple)
    Output:
        components: Numpy array of shape (num_concepts, Representation_dim).
        Stores underlying dictionary elements / concept vectors
        comp_activ: Numpy array of shape (N_samples, num_concepts).
        Activations/Coefficients matrix.
        model: sklearn dictionary learning / clustering object.

    This function performs dictionary learning for given representation matrix
    and extracts concept vectors and their activations.
    """

    assert num_concepts is not None, "Number of components is None!"
    assert (
        decomposition_method is not None
    ), "Decomposition method specified is None!"

    if torch.is_tensor(mat):
        # Convert to numpy array for sklearn processing
        if mat.dtype == torch.bfloat16:
            mat = mat.to(torch.float32)
        mat = mat.cpu().numpy()
    # mat -= np.min(mat)
    mat = mat.astype(np.float32)
    print("Matrix stats:")
    print("Min:", np.min(mat), "Max:", np.max(mat))
    print("NaNs:", np.isnan(mat).sum(), "Infs:", np.isinf(mat).sum())
    print("Shape:", mat.shape, "Rank:", np.linalg.matrix_rank(mat))
    assert (
        len(mat.shape) == 2
    ), f"Given feature matrix needs to be 2D. Shape encountered {mat.shape}"

    if decomposition_method == "pca":
        model = PCA(n_components=num_concepts)
        comp_activ = model.fit_transform(mat)
        components = model.components_
    elif decomposition_method == "snmf":
        print(
            "Using Semi-NMF / Non-negative dictionary learning with alpha:",
            args.dl_alpha,
        )
        model = DictionaryLearning(
            n_components=num_concepts,
            positive_code=True,
            fit_algorithm="cd",
            alpha=args.dl_alpha,
            transform_algorithm="lasso_cd",
            max_iter=args.dl_max_iter,
            # The inner lasso (codes) stopped at sklearn's 1000 iterations
            # without converging for K >= 4; same concepts, no warnings.
            transform_max_iter=max(1000, getattr(args, "dl_max_iter", 1000)),
        )
        comp_activ = model.fit_transform(mat)
        components = model.components_
        components, comp_activ = l1_sparse_atoms(
            mat,
            components,
            comp_activ,
            code_alpha=model.alpha,
            dict_alpha=getattr(args, "dl_dict_alpha", 0.0),
        )
    elif decomposition_method == "simple":
        model = KMeans(n_clusters=num_concepts, max_iter=args.dl_max_iter)
        norms = np.linalg.norm(mat, axis=1)
        indices = norms.argsort()[-num_concepts:]
        components = mat[indices]
        model.cluster_centers_ = components + 0
        # Kmeans transforms to cluster distances and not "activations". 1/(1+x)
        # transformation to view distances as activations
        comp_activ = 1 / (1 + model.transform(mat))
    elif decomposition_method == "sae2":
        # DEVICE from .env is the source of truth (get_device_config reads it).
        device = get_device_config(None).primary_device

        input_dim = mat.shape[1]
        hidden_dim = num_concepts
        sae_sparsity_lambda = getattr(args, "sae_sparsity_lambda", 0.0005)
        sae_top_k = _compute_sae_top_k(args, hidden_dim)
        if sae_top_k is not None:
            print(
                "Using SAE2 with hard top-k activation: "
                f"top_k={sae_top_k}/{hidden_dim} "
                f"(target_sparsity={args.sae_target_sparsity})"
            )
        else:
            print("Using SAE2 with sparsity_lambda:", sae_sparsity_lambda)

        model = SparseAutoencoder(
            input_dim,
            hidden_dim,
            sparsity_lambda=sae_sparsity_lambda,
            top_k=sae_top_k,
        ).to(device)

        with torch.enable_grad():
            train_sae(model, mat, epochs=5000, device=device)
        model.eval()
        with torch.no_grad():
            mat_tensor = torch.tensor(mat, dtype=torch.float32, device=device)
            _, sparse_codes = model(mat_tensor)
        comp_activ = sparse_codes.cpu().numpy()
        # decoder.weight is (dim, concepts): concepts are its columns.
        components = model.decoder.weight.detach().cpu().numpy().T

    else:
        raise ValueError(
            f"unknown decomposition_method {decomposition_method!r}"
        )

    return components, comp_activ, None  # Optionally return model
