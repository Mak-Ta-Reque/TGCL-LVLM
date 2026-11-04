#!/usr/bin/env python3
"""Explain an existing run's answers with a different concept bank, without the
VLM.

The explainer's answers and per-token hidden states
(vlm_explanations_activations.npy)
do not depend on the concept bank; only the concept ranking does. This rebuilds
every
token's top concepts (cosine to the bank, as
inference/vlm_explainer_multibatch.py) and
the sequence-level ones (mean of the tokens kept by keep_token_text) from the
stored
activations, so a new bank is evaluated in minutes instead of a full explainer
run.

    python scripts/rerank_explanations.py \\
        --run outputs/paper/tgcl_sam_coco_seed1 --method snmf \\
        --bank new_bank.pth --out outputs/paper/tgcl_sam_coco_posthoc_seed1

The new run directory gets concept/<method>/<bank name>, explanations*/<method>/
for every
evaluation variant of the source run, and links to the source's features/ and
inference/,
so pipeline steps 7-8 (--skip-to-step 7) and scripts/paper_report.py run on it
unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from tgcl.helpers.explanations import load
from tgcl.helpers.explanations import TABLE_KEY
from tgcl.helpers.image_refs import image_classes
from tgcl.helpers.run_layout import find_concept_bank
from tgcl.inference.vlm_explainer_multibatch import keep_token_text


def unit_concepts(bank: Dict[str, Any], centred: bool = False) -> np.ndarray:
    """Unit-norm concepts; with centring, minus the bank mean first (as
    VLMConceptExplainer.set_centering)."""
    c = np.asarray(
        torch.as_tensor(bank["concepts"]).float().numpy(), dtype=np.float64
    )
    if centred:
        c = c - c.mean(axis=0, keepdims=True)
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)


def ranked(vec: np.ndarray, unit: np.ndarray, n: int) -> List[Dict[str, Any]]:
    """The n concepts most cosine-similar to vec (unit: unit concepts)."""
    v = np.asarray(vec, dtype=np.float64)
    sims = unit @ (v / max(np.linalg.norm(v), 1e-8))
    order = np.argsort(-sims, kind="stable")[:n]
    return [
        {
            "rank": i + 1,
            "concept_index": int(c),
            "distance": float(1.0 - sims[c]),
            "similarity": float(sims[c]),
        }
        for i, c in enumerate(order)
    ]


def concept_row(bank: Dict[str, Any], idx: int) -> Dict[str, Any]:
    def pick(key):
        values = bank.get(key)
        return values[idx] if values is not None and idx < len(values) else None

    return {
        "concept_name": pick("concept_names"),
        "text_grounding": pick("text_grounding"),
        "image_grounding_path": pick("image_grounding_paths"),
        "image_grounding_bboxes": pick("image_grounding_bboxes"),
    }


def rerank(
    payload: Dict[str, Any],
    acts: np.ndarray,
    bank: Dict[str, Any],
    top_n: int,
    center: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """payload with every top-concept list recomputed from the stored
    activations.

    Args:
        payload: The explanations (results with per-token entries).
        acts: The stored raw answer states (activation_index rows).
        bank: The concept bank to rank against.
        top_n: Concepts kept per token.
        center: The explainer's calibration mean (centred matching): when
            given, it is subtracted from every state and the bank mean from
            every concept, as in the explainer run.
    """
    unit = unit_concepts(bank, centred=center is not None)
    shift = 0.0 if center is None else np.asarray(center, dtype=np.float64)
    n = min(top_n, len(unit))
    used = set()
    for r in payload["results"]:
        tokens = r.get("per_token_concepts") or []
        for t in tokens:
            t["top_concepts"] = ranked(
                acts[t["activation_index"]] - shift, unit, n
            )
            used.update(c["concept_index"] for c in t["top_concepts"])
        if tokens:
            kept = [
                t
                for t in tokens
                if keep_token_text(str(t.get("token_text", "")))
            ] or tokens
            pooled = np.mean(
                [acts[t["activation_index"]] for t in kept], axis=0
            )
            r["top_concepts_over_sequence"] = ranked(pooled - shift, unit, n)
            used.update(
                c["concept_index"] for c in r["top_concepts_over_sequence"]
            )
    payload[TABLE_KEY] = {str(i): concept_row(bank, i) for i in sorted(used)}
    return payload


def first_per_class(results: List[Dict[str, Any]], n: int) -> List[Dict]:
    """The first n results of every image class (in file order)."""
    seen: Dict[Optional[str], int] = {}
    kept = []
    for r in results:
        label = (image_classes(r.get("image_path", "")) or [None])[0]
        if seen.get(label, 0) < n:
            seen[label] = seen.get(label, 0) + 1
            kept.append(r)
    return kept


def rerank_file(
    json_path: Path,
    bank: Dict[str, Any],
    out_json: Path,
    top_n: int,
    centred: bool = True,
    per_class: int = 0,
) -> int:
    """Re-rank one explanation file against another concept bank.

    The stored activations are scored against bank; out_json gets the new top_n
    concepts per token (the old grounding is dropped) and a symlink to the
    activations.

    Args:
        json_path: The source explanations.
        bank: The concept bank to rank against.
        out_json: The new explanation file.
        top_n: Concepts kept per token.
        centred: Centred matching when the source saved its calibration mean
            (False: plain cosine, as an explainer run without EXPL_CENTER).
        per_class: Keep only the first per_class images of every class (0:
            all).

    Returns:
        The number of explained images.
    """
    payload = load(json_path)
    if per_class > 0:
        payload["results"] = first_per_class(payload["results"], per_class)
    # The states next to the JSON win over its stored (possibly foreign) path.
    sibling = json_path.with_name(json_path.stem + "_activations.npy")
    acts_path = (
        str(sibling) if sibling.exists() else payload.get("activations_path")
    )
    acts = np.load(acts_path, mmap_mode="r")
    for r in payload[
        "results"
    ]:  # drop the old bank's grounding: the new table replaces it
        for t in r.get("per_token_concepts") or []:
            t["top_concepts"] = []
        r["top_concepts_over_sequence"] = []
    payload.pop(TABLE_KEY, None)
    # Centred matching (explainer --center_calibration) saves its mean here.
    center_path = json_path.with_name(json_path.stem + "_center.npy")
    center = np.load(center_path) if centred and center_path.exists() else None
    rerank(payload, acts, bank, top_n, center)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if center is not None:
        np.save(out_json.with_name(out_json.stem + "_center.npy"), center)
    os.symlink(
        os.path.abspath(acts_path),
        out_json.with_name("vlm_explanations_activations.npy"),
    )
    payload["activations_path"] = str(
        out_json.with_name("vlm_explanations_activations.npy")
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return len(payload["results"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run", type=Path, required=True, help="source run directory"
    )
    ap.add_argument("--method", default="snmf")
    ap.add_argument(
        "--bank", type=Path, required=True, help="new concept bank (.pth)"
    )
    ap.add_argument("--out", type=Path, required=True, help="new run directory")
    ap.add_argument("--top_n", type=int, default=3)
    ap.add_argument(
        "--no-center",
        action="store_true",
        help="plain cosine matching, as an explainer run without EXPL_CENTER",
    )
    ap.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="keep the first N images of every class (default: all)",
    )
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new run directory")
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    src_bank = find_concept_bank(args.run, args.method)
    bank_dst = args.out / "concept" / args.method / src_bank.name
    bank_dst.parent.mkdir(parents=True)
    shutil.copyfile(args.bank, bank_dst)
    for sub in ("features", "inference"):
        if (args.run / sub).exists():
            os.symlink((args.run / sub).resolve(), args.out / sub)
    for exp_dir in sorted(args.run.glob("explanations*")):
        src = exp_dir / args.method / "vlm_explanations.json"
        if src.exists():
            n = rerank_file(
                src,
                bank,
                args.out / exp_dir.name / args.method / "vlm_explanations.json",
                args.top_n,
                centred=not args.no_center,
                per_class=args.per_class,
            )
            print(f"{exp_dir.name}: {n} results re-ranked", flush=True)


if __name__ == "__main__":
    main()
