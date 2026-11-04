#!/usr/bin/env python3
"""Concept insertion/deletion curves: the paper's faithfulness metric.

For every explained test image, h is the stored hidden state of the answer's
first token (vlm_explanations_activations.npy, written by the explainer) and
the bank's K concepts are ordered by their activation on h, exactly as the
explainer ranked them (cosine; centred on the explainer's calibration mean and
the bank mean when the explanation was centred). With P_m the projection onto
the span of the first m concepts of that order (QR of the ordered bank):

    deletion   h_m = h - P_m h                 m = 0 ... K
    insertion  h_m = (h - P_K h) + P_m h

Each state goes through the model's output path ([final norm] -> lm_head ->
logit soft-cap) and the curve is the probability of the token the model
generated, divided by p(h). Both curves join the full state (1) and the
concept-free state h - P_K h. On the fraction-of-concepts axis x = m / K:

    gap     = area(insertion) - area(reversed deletion)
              (0: the concepts do not carry the prediction; up to 1)
    random  the same gap with the concepts in random order (the control)
    net     gap - random: what the concepts' order adds; removes effects any
            direction of the bank has (e.g. one all answer states share)

    python -m tgcl.evaluation.concept_insertion_deletion \\
        --results_json <run>/explanations/snmf/vlm_explanations.json \\
        --concept_path <run>/concept/snmf/combined_concept_snmf_cr0.6_raw.pth \\
        --out_dir <run>/eval/snmf

writes concept_curves.json (the summary) and concept_curves.npz (per-image
curves) to --out_dir.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tgcl.device_utils import get_device_config
from tgcl.helpers.explanations import load
from tgcl.models import get_model_class

# Written by the explainer into vlm_explanations.json; files without it store
# token-axis-normalized activations and cannot be evaluated.
ACTIVATIONS_KIND = "hidden_state"
FINAL_NORM_PATHS = (
    "model.language_model.norm",
    "model.norm",
    "language_model.model.norm",
    "model.text_model.norm",
    "transformer.ln_f",
)
SUMMARY = "concept_curves.json"
CURVES = "concept_curves.npz"
ORDERS = ("value", "random")


# ------------------------------------------------------------------ geometry


def concept_grid(k: int, n_log: int = 24, dense: int = 10) -> np.ndarray:
    """Numbers of concepts m on the curve: every m up to `dense`, then
    log-spaced up to k (always 0 and k)."""
    steps = {0, k} | set(range(min(k, dense) + 1))
    steps |= {int(round(x)) for x in np.geomspace(1, max(k, 1), n_log)}
    return np.array(sorted(m for m in steps if 0 <= m <= k), dtype=np.int64)


def unit_rows(c: torch.Tensor) -> torch.Tensor:
    """Rows scaled to unit L2 norm."""
    return c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)


def ranking_concepts(concepts: torch.Tensor, centred: bool) -> torch.Tensor:
    """The unit directions the explainer ranks with (minus the bank mean when
    centred, as VLMConceptExplainer.set_centering)."""
    c = unit_rows(concepts)
    return unit_rows(c - c.mean(0)) if centred else c


def ordered_states(
    h: torch.Tensor, concepts: torch.Tensor, order: torch.Tensor, ms
) -> tuple:
    """(deletion, insertion) states, [len(ms), D] each: h without, and the
    concept-free part of h with, the span of the first m concepts of order."""
    q, _ = torch.linalg.qr(concepts[order].T)  # D x K, nested ordered spans
    parts = q * (q.T @ h)  # D x K: h's part along each new direction
    cum = torch.cat([torch.zeros_like(h)[:, None], parts.cumsum(1)], 1)
    cum = cum[:, torch.as_tensor(ms, device=h.device)]
    deletion = (h[:, None] - cum).T
    insertion = (h[:, None] - cum[:, -1:] + cum).T
    return deletion, insertion


def area(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid area under y(x) (x in any order)."""
    o = np.argsort(x)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(np.asarray(y)[o], np.asarray(x)[o]))


def curve_gap(x: np.ndarray, insertion: np.ndarray, deletion: np.ndarray):
    """area(insertion) - area(reversed deletion) on x = fraction of
    concepts (curves relative to the full state)."""
    return area(x, insertion) - area(1.0 - x, deletion)


# --------------------------------------------------------------- evaluation


def explained_states(payload: Dict[str, Any], acts: np.ndarray):
    """(image, generated token id, hidden state) of every explained image's
    first answer token (the token Acc / F1 score)."""
    for item in payload.get("results", []):
        tokens = item.get("per_token_concepts") or []
        if not tokens:
            continue
        token_id = int(tokens[0].get("token_id", -1))
        index = tokens[0].get("activation_index")
        if token_id < 0 or index is None:
            continue
        yield item.get("image_path", ""), token_id, acts[int(index)]


def evaluate_payload(
    payload: Dict[str, Any],
    acts: np.ndarray,
    concepts: torch.Tensor,
    token_probs: Callable[[torch.Tensor, int], torch.Tensor],
    center: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Concept curves of every explained image.

    Args:
        payload: The explanations (results with per-token entries).
        acts: The stored hidden states (activation_index rows).
        concepts: The bank, [K, D].
        token_probs: (states [N, D], token id) -> probability of the token
            under the model's output path, [N].
        center: The explainer's calibration mean: given, the ranking is
            centred as in the explainer run.
        seed: Seed of the random orders.

    Returns:
        {"summary": gap / random / net / AUCs / n / K, "curves": arrays}.
    """
    device = concepts.device
    c = unit_rows(concepts.double())
    rank = ranking_concepts(concepts.double(), center is not None)
    shift = torch.as_tensor(
        center if center is not None else np.zeros(c.shape[1]),
        dtype=torch.float64,
        device=device,
    )
    k = len(c)
    ms = concept_grid(k)
    x = ms / k
    generator = torch.Generator().manual_seed(seed)
    curves: Dict[str, list] = {"images": [], "p_full": []}
    for order_name in ORDERS:
        curves[f"del_{order_name}"], curves[f"ins_{order_name}"] = [], []
    for image, token_id, state in explained_states(payload, acts):
        h = torch.tensor(np.array(state), dtype=torch.float64, device=device)
        orders = {
            "value": torch.argsort(rank @ (h - shift), descending=True),
            "random": torch.randperm(k, generator=generator).to(device),
        }
        states = [h[None]]
        for order_name in ORDERS:
            states.extend(ordered_states(h, c, orders[order_name], ms))
        p = token_probs(torch.cat(states), token_id).double().cpu().numpy()
        p_full = max(float(p[0]), 1e-30)
        rel = p[1:] / p_full
        n = len(ms)
        for i, order_name in enumerate(ORDERS):
            curves[f"del_{order_name}"].append(rel[2 * i * n : (2 * i + 1) * n])
            curves[f"ins_{order_name}"].append(
                rel[(2 * i + 1) * n : (2 * i + 2) * n]
            )
        curves["images"].append(image)
        curves["p_full"].append(p_full)
    arrays = {key: np.asarray(v) for key, v in curves.items()}
    arrays["ms"], arrays["k"] = ms, np.asarray(k)
    gaps = {
        o: [
            curve_gap(x, i, d)
            for i, d in zip(arrays[f"ins_{o}"], arrays[f"del_{o}"])
        ]
        for o in ORDERS
    }
    n_images = len(arrays["images"])
    nan = float("nan")
    summary = {
        "n_images": n_images,
        "n_concepts": k,
        "gap": float(np.mean(gaps["value"])) if n_images else nan,
        "gap_random": float(np.mean(gaps["random"])) if n_images else nan,
        "insertion_auc": (
            float(np.mean([area(x, y) for y in arrays["ins_value"]]))
            if n_images
            else nan
        ),
        "deletion_auc": (
            float(np.mean([area(x, y) for y in arrays["del_value"]]))
            if n_images
            else nan
        ),
        "centred": center is not None,
        "seed": seed,
    }
    summary["gap_net"] = summary["gap"] - summary["gap_random"]
    return {"summary": summary, "curves": arrays}


def write_outputs(result: Dict[str, Any], out_dir: Union[str, Path]) -> Path:
    """concept_curves.json and concept_curves.npz in out_dir; returns the
    summary's path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / CURVES, **result["curves"])
    path = out_dir / SUMMARY
    path.write_text(json.dumps(result["summary"], indent=1), encoding="utf-8")
    return path


def read_summary(out_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """The concept_curves.json of an evaluation folder, or None."""
    path = Path(out_dir) / SUMMARY
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------------------------------------------------------- model


def _get_module(root: torch.nn.Module, path: str) -> Optional[torch.nn.Module]:
    cur = root
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur if isinstance(cur, torch.nn.Module) else None


class OutputHead(torch.nn.Module):
    """The model's path from the hooked layer to logits: [final norm] ->
    lm_head -> soft-cap. The final norm is applied only when the hooked layer
    is not the final norm itself (a logit-lens readout of an earlier layer)."""

    def __init__(self, model: torch.nn.Module, layer_path: str) -> None:
        super().__init__()
        self.lm_head = model.get_output_embeddings()
        if self.lm_head is None:
            raise RuntimeError("Model has no output embeddings (lm_head)")
        self.norm = None
        for path in FINAL_NORM_PATHS:
            module = _get_module(model, path)
            if module is not None:
                if path != layer_path:
                    self.norm = module
                break
        text_config = (
            model.config.get_text_config()
            if hasattr(model.config, "get_text_config")
            else model.config
        )
        self.softcap = getattr(text_config, "final_logit_softcapping", None)
        self.dtype = next(self.lm_head.parameters()).dtype
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.to(self.dtype)
        if self.norm is not None:
            x = self.norm(x)
        logits = self.lm_head(x).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


class ConceptCurveEvaluator:
    """Loads the VLM's output path once and evaluates explanation files.

    Args:
        model_name: The VLM (the explainer's).
        layer_path: The hooked layer of the explanations.
        device: Device (default: DEVICE).
        batch: States per output-head call.
    """

    def __init__(
        self,
        model_name: str,
        layer_path: str = "model.language_model.norm",
        device: Optional[str] = None,
        batch: int = 512,
    ) -> None:
        for key, value in (
            ("TF_CPP_MIN_LOG_LEVEL", "3"),
            ("TRANSFORMERS_VERBOSITY", "error"),
            ("HF_HUB_DISABLE_TELEMETRY", "1"),
        ):
            os.environ.setdefault(key, value)
        device_config = get_device_config(device)
        model_class = get_model_class(
            model_name_or_path=model_name,
            processor_name=model_name,
            device=device_config.primary_device,
            logger=None,
            args=argparse.Namespace(local_files_only=False, cache_dir=None),
            device_config=device_config,
        )
        self.head = OutputHead(model_class.get_model().eval(), layer_path)
        self.device = next(self.head.lm_head.parameters()).device
        self.batch = batch

    def token_probs(self, states: torch.Tensor, token_id: int) -> torch.Tensor:
        """Probability of token_id for every state."""
        out = []
        with torch.no_grad():
            for i in range(0, len(states), self.batch):
                logits = self.head(states[i : i + self.batch].to(self.device))
                out.append(F.softmax(logits, dim=-1)[:, token_id].float())
        return torch.cat(out)

    def evaluate(
        self,
        results_json: Union[str, Path],
        concept_path: Union[str, Path],
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Concept curves of one explanation file against a concept bank
        (centred when the explainer saved its calibration mean next to it)."""
        results_json = Path(results_json)
        payload = load(results_json)
        if payload.get("activations_kind") != ACTIVATIONS_KIND:
            raise ValueError(
                f"{results_json} has no raw per-token activations; re-run "
                "pipeline step 6."
            )
        acts_path = results_json.with_name(
            results_json.stem + "_activations.npy"
        )
        if not acts_path.exists():
            acts_path = Path(payload.get("activations_path", ""))
        acts = np.load(acts_path, mmap_mode="r")
        center_path = results_json.with_name(results_json.stem + "_center.npy")
        center = np.load(center_path) if center_path.exists() else None
        bank = torch.load(concept_path, map_location="cpu", weights_only=False)
        concepts = torch.as_tensor(bank["concepts"]).to(self.device)
        if concepts.shape[1] != acts.shape[1]:
            raise ValueError(
                f"concept dim {concepts.shape[1]} != state dim {acts.shape[1]}"
            )
        return evaluate_payload(
            payload, acts, concepts, self.token_probs, center, seed
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--concept_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--model_name",
        default=os.environ.get("VLM_MODEL", "google/gemma-3n-E4B-it"),
    )
    ap.add_argument("--layer_path", default=None, help="default: from JSON")
    ap.add_argument("--device", default=None, help="cuda, cpu or cuda:N")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    layer = args.layer_path or load(args.results_json).get(
        "layer_path", "model.language_model.norm"
    )
    evaluator = ConceptCurveEvaluator(args.model_name, layer, args.device)
    result = evaluator.evaluate(args.results_json, args.concept_path, args.seed)
    path = write_outputs(result, args.out_dir)
    s = result["summary"]
    print(
        f"{s['n_images']} images, {s['n_concepts']} concepts: gap "
        f"{s['gap']:.3f}, random {s['gap_random']:.3f}, net "
        f"{s['gap_net']:.3f} -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
