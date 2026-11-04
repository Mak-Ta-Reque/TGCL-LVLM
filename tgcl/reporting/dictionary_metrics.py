"""Concept-quality metrics of a concept bank W (K concepts x D) for the paper's
table (scripts/dictionary_metrics.py, scripts/dictionary_table.py):

Sparsity ↑          gini_sparsity: how concentrated each concept vector is
                    over the hidden dimensions
Overlap ↓           l2_overlap: mean |cos| between concepts
Instability ↓       instability: 1 - matched cosine between seeds' banks
"""

from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Coordinates below this fraction of a concept's L2 norm are dropped before
# the overlap.
OVERLAP_THRESHOLD = 0.01

# The paper's rows: method x decomposition. TGCL is semi-NMF combined by
# symmetric alignment (Lowdin orthogonalization) and noise removal (each concept
# keeps its 50 % largest coordinates); every other row is evaluated on its
# concepts as learned. Other combinations: add them here (one row each).
METHODS = ["CRAFT", "MCD", "CoX-LMM", "TGCL (SAM)", "TGCL (sliding window)"]
DECOMPS = [("snmf", "SNMF"), ("sae2", "SAE"), ("simple", "SIMPLE")]
TGCL_COMBINATIONS = {"lowdin+magnitude:0.5": "SNMF"}


def is_tgcl(method: str) -> bool:
    """Whether a method name is one of TGCL's settings."""
    return method.startswith("TGCL")


def posts_of(method: str, decomposition: str) -> List[str]:
    """The post-processings a (method, decomposition) row is reported with."""
    if is_tgcl(method) and decomposition == "snmf":
        return list(TGCL_COMBINATIONS)
    return ["none"]


def table_rows() -> List[Tuple[str, str, str, str]]:
    """(method, decomposition, row label, post-processing) in table order."""
    rows = []
    for method in METHODS:
        for key, label in DECOMPS:
            for post in posts_of(method, key):
                rows.append(
                    (method, key, TGCL_COMBINATIONS.get(post, label), post)
                )
    return rows


def unit_rows(W: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm (zero rows stay zero)."""
    W = np.asarray(W, dtype=np.float64)
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)


def gini_sparsity(W: np.ndarray) -> float:
    """Gini sparsity of the concept vectors (Hurley & Rickard 2009), mean
    over concepts: 0 = all coordinates equal, 1 - 1/D = one non-zero
    coordinate. Scale-free."""
    a = np.sort(np.abs(np.asarray(W, dtype=np.float64)), axis=1)
    n = a.shape[1]
    weights = (n - np.arange(1, n + 1) + 0.5) / n
    shares = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12)
    return float(np.mean(1 - 2 * shares @ weights))


def l2_overlap(W: np.ndarray, threshold: float = OVERLAP_THRESHOLD) -> float:
    """Mean |cos| between different concepts after dropping coordinates with
    |w_i| / ||w||_2 < threshold (0 = orthogonal). Scale- and sign-free; NaN
    for fewer than two concepts."""
    u = unit_rows(W)
    u = unit_rows(np.where(np.abs(u) < threshold, 0.0, u))
    if len(u) < 2:
        return float("nan")
    sims = np.abs(u @ u.T)
    return float(sims[~np.eye(len(u), dtype=bool)].mean())


def matched_cosine(W1: np.ndarray, W2: np.ndarray) -> float:
    """Mean cosine of the one-to-one matching (Hungarian) between two
    banks."""
    sims = unit_rows(W1) @ unit_rows(W2).T
    rows, cols = linear_sum_assignment(-sims)
    return float(sims[rows, cols].mean())


def instability(banks: Sequence[np.ndarray]) -> float:
    """1 - matched cosine, averaged over all pairs of banks (seeds); NaN for
    fewer than two banks. Lower = the concepts are found again."""
    pairs = list(combinations(banks, 2))
    if not pairs:
        return float("nan")
    return float(np.mean([1 - matched_cosine(a, b) for a, b in pairs]))
