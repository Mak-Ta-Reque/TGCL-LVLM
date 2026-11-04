"""Metrics behind the paper tables (scripts/paper_report.py,
scripts/intermediates.py).

Concept banks       load_concepts, concept_overlap, matched_cosine_similarity
(instability)
Concept quality     hoyer_selectivity, near_zero_fraction, gini_index,
effective_coordinates,
                    effective_rank_fraction, score_correlation
Object identity     canonical_label, load_predictions_from_json, compute_metrics
(Obj-F1)
"""

from collections import Counter
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from tgcl.helpers.explanations import expand
from tgcl.helpers.image_refs import image_classes

# ---------------------------------------------------------------------- concept
# banks


def _orient_atoms(C: torch.Tensor) -> torch.Tensor:
    return C if C.size(0) <= C.size(1) else C.T.contiguous()


def load_concepts(path: Path) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    C = torch.as_tensor(blob["concepts"], dtype=torch.float32)
    if C.ndim != 2 or C.numel() == 0:
        # Degenerate/empty bank (e.g. a config where 0 concepts passed the
        # CLEAN_EXAMPLE_RATIO purity filter still gets an empty file saved to
        # disk, shape torch.Size([0])) -- report as missing, don't crash.
        return None
    return _orient_atoms(C)


def hoyer_selectivity(C: torch.Tensor) -> float:
    """Higher = more concentrated/sparse atoms."""
    eps = 1e-12
    n = C.size(1)
    sqrt_n = torch.sqrt(torch.tensor(float(n)))
    l1 = C.abs().sum(dim=1)
    l2 = torch.linalg.norm(C, dim=1) + eps
    s = (sqrt_n - (l1 / l2)) / (sqrt_n - 1.0 + eps)
    return s.clamp_(0.0, 1.0).mean().item()


# |c_i| below this on the unit-norm concept counts as zero (near-zero fraction)
# and is dropped before the
# cosine overlap. 0.1 (user choice, 2026-10-27): only a concept's largest
# coordinates count (~10 of 2048-3584).
NEAR_ZERO_THRESHOLD = 0.1


def concept_overlap(
    C: torch.Tensor, threshold: float = NEAR_ZERO_THRESHOLD
) -> float:
    """Mean off-diagonal cosine similarity between atoms. Lower = more
    disentangled.

    Coordinates with |c_i| / ||c||_2 < threshold are dropped first (relative
    to each atom's L2 norm, as near_zero_fraction), so the result does not
    depend on the bank's scale (simple and sae2 atoms are not unit-norm).
    """
    C_thr = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C_thr[C_thr.abs() < threshold] = 0.0
    Cn = C_thr / (C_thr.norm(dim=1, keepdim=True) + 1e-12)
    sim = Cn @ Cn.T
    n = sim.size(0)
    if n <= 1:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim[mask].mean().item()


def matched_cosine_similarity(
    V1: np.ndarray, V2: np.ndarray, sign_invariant: bool = False
) -> float:
    """Mean matched cosine similarity between two atom banks via Hungarian
    matching on the full rectangular similarity matrix (no truncation).

    sign_invariant compares |cos|: for PCA, whose components are defined only
    up to sign, a flipped but identical component is the same concept.
    """
    V1n = V1 / (np.linalg.norm(V1, axis=1, keepdims=True) + 1e-12)
    V2n = V2 / (np.linalg.norm(V2, axis=1, keepdims=True) + 1e-12)
    S = V1n @ V2n.T
    if sign_invariant:
        S = np.abs(S)
    row_ind, col_ind = linear_sum_assignment(-S)
    return float(S[row_ind, col_ind].mean())


# ---------------------------------------------------------------------- concept
# quality


def near_zero_fraction(C, threshold: float = NEAR_ZERO_THRESHOLD) -> float:
    """Fraction of coordinates with |c_i| / ||c||_2 < threshold, mean over
    concepts (for unit-norm
    banks such as snmf/pca: the share of coordinates below the absolute
    threshold).
    """
    C = C.double()
    unit = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (unit.abs() < threshold).double().mean(dim=1).mean().item()


def _unit_rows(C) -> np.ndarray:
    c = (
        C.double().numpy()
        if hasattr(C, "double")
        else np.asarray(C, dtype=np.float64)
    )
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)


def gini_index(C) -> float:
    """Gini sparsity (Hurley & Rickard 2009) per concept, mean: 0 = all
    coordinates equal, 1 = one nonzero."""
    a = np.sort(np.abs(_unit_rows(C)), axis=1)
    n = a.shape[1]
    weights = (n - np.arange(1, n + 1) + 0.5) / n
    return float(np.mean(1 - 2 * (a / a.sum(1, keepdims=True)) @ weights))


def effective_coordinates(C) -> float:
    """Effective number of coordinates per concept, 1 / sum p_i^2 with p = c_i^2
    (unit c), mean."""
    p = _unit_rows(C) ** 2
    return float(np.mean(1.0 / np.sum(p**2, axis=1)))


def effective_rank_fraction(C) -> float:
    """exp(entropy of the normalised squared singular values) / K of the unit
    concepts (Roy & Vetterli 2007):
    the share of the K concepts that are independent directions (1 = orthogonal,
    1/K = all identical).
    """
    u = _unit_rows(C)
    s = np.linalg.svd(u, compute_uv=False)
    p = s**2 / np.sum(s**2)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))) / len(u))


def score_correlation(C, states: np.ndarray) -> Optional[float]:
    """Mean |Pearson correlation| between two concepts' scores (cosine numerator
    h.c) over hidden states:
    functional overlap, i.e. do the concepts respond to the same tokens. None
    for < 2 concepts.
    """
    u = _unit_rows(C)
    if len(u) < 2 or states is None or len(states) < 3:
        return None
    corr = np.corrcoef((np.asarray(states, dtype=np.float64) @ u.T).T)
    return float(np.nanmean(np.abs(corr[~np.eye(len(u), dtype=bool)])))


# ---------------------------------------------------------------------- object
# identity (Obj-F1)


def canonical_label(label: Optional[str]) -> Optional[str]:
    """Case- and separator-insensitive class name: folder "garbage_truck" and
    concept tag "garbage truck" compare equal."""
    if label is None:
        return None
    return " ".join(str(label).replace("_", " ").lower().split())


def concept_label(concept_name: Optional[List[str]]) -> Optional[str]:
    if not concept_name:
        return None
    return canonical_label(Counter(concept_name).most_common(1)[0][0])


def load_predictions_from_json(
    path: Path,
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """[(true_label, pred_rank1, pred_rank2), ...] from an explainer output.

    The true label is the image's class (helpers.image_refs); predictions are
    the rank-1/2 concepts of the first answer token; all labels canonical
    (canonical_label). [] if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        d = expand(json.load(f))
    out = []
    for item in d.get("results", []):
        true_label = canonical_label(
            (image_classes(item["image_path"]) or [None])[0]
        )
        ptc = item.get("per_token_concepts") or []
        if not ptc:
            out.append((true_label, None, None))
            continue
        top_concepts = ptc[0].get("top_concepts") or []
        by_rank = {c["rank"]: c for c in top_concepts}
        pred1 = concept_label(by_rank.get(1, {}).get("concept_name"))
        pred2 = concept_label(by_rank.get(2, {}).get("concept_name"))
        out.append((true_label, pred1, pred2))
    return out


def _f1(tp: int, fp: int, fn: int) -> float:
    """F1 from counts; 0 when undefined (sklearn's zero_division=0)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def _top1_f1_macro(
    y_true: List[str], pred1: List[Optional[str]], classes: List[str]
) -> float:
    """Macro F1 over `classes` with one predicted label per image -- equal to
    sklearn.metrics.f1_score(average="macro", labels=classes, zero_division=0).
    """
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, pred1) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, pred1) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, pred1) if t == c and p != c)
        f1s.append(_f1(tp, fp, fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def _top2_f1_macro(
    y_true: List[str],
    pred1: List[Optional[str]],
    pred2: List[Optional[str]],
    classes: List[str],
) -> float:
    """Macro F1 under the containment criterion (class among the two guesses),
    over the classes present in y_true."""
    f1s = []
    for c in classes:
        tp = fp = fn = 0
        for t, p1, p2 in zip(y_true, pred1, pred2):
            guessed = c in (p1, p2)
            if t == c:
                if guessed:
                    tp += 1
                else:
                    fn += 1
            elif guessed:
                fp += 1
        if tp + fn == 0:
            continue  # class not present in this config's eval set
        f1s.append(_f1(tp, fp, fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def compute_metrics(
    rows: List[Tuple[str, Optional[str], Optional[str]]],
    classes: Optional[List[str]] = None,
) -> dict:
    """Object recognition from the top concepts' classes (Obj-F1).

    Args:
        rows: (true class, top-1 predicted class, top-2 predicted class) per
            image; predictions may be None.
        classes: Classes of the macro F1 (default: those in the ground truth).

    Returns:
        n_images, top1_accuracy, top1_f1_macro, top2_accuracy, top2_f1_macro.
    """
    classes = [
        canonical_label(c)
        for c in (sorted({r[0] for r in rows}) if classes is None else classes)
    ]
    y_true = [r[0] for r in rows]
    pred1 = [r[1] for r in rows]
    pred2 = [r[2] for r in rows]

    top1_hits = [t == p1 for t, p1 in zip(y_true, pred1)]
    top2_hits = [h or (t == p2) for h, t, p2 in zip(top1_hits, y_true, pred2)]

    top1_f1 = _top1_f1_macro(y_true, pred1, classes)
    top2_f1 = _top2_f1_macro(y_true, pred1, pred2, classes)

    return {
        "n_images": len(rows),
        "top1_accuracy": float(np.mean(top1_hits)),
        "top1_f1_macro": float(top1_f1),
        "top2_accuracy": float(np.mean(top2_hits)),
        "top2_f1_macro": float(top2_f1),
    }
