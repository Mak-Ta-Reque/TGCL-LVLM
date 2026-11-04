#!/usr/bin/env python3
"""Concept-quality metrics of finished runs' concept banks (no VLM needed).

For every spec row (one method on one dataset, one run per seed) this loads
each seed's concept bank, as learned and as TGCL combines it (Lowdin
orthogonalization and keeping 50 % of each concept's coordinates), and
reports
Gini sparsity, L2 overlap and instability (tgcl/reporting/dictionary_metrics.py)
per post-processing. scripts/dictionary_table.py turns the CSV into the paper's
table.

    python scripts/dictionary_metrics.py \\
        --spec configs/dictionary_metrics/gemma3n_coco10.json \\
        --out outputs/dictionary_metrics/gemma3n_coco10_concepts.csv

Spec (JSON): {"rows": [{"dataset": ..., "method": ..., "decomposition":
"snmf"|"sae2"|"simple", "all_components": bool, "runs": [run dir per seed]}]}.
all_components also counts the directions combine_concepts moved to the
negative bank (banks built before every direction was kept without a purity
filter). Run paths are relative to the repository root.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from sparsify_concepts import lowdin_orthogonalize
from sparsify_concepts import magnitude_sparsify
import torch

from tgcl.reporting import dictionary_metrics as dm

ROOT = Path(__file__).resolve().parent.parent
# Every bank as learned, and as TGCL combines it.
POSTS = ("none", *dm.TGCL_COMBINATIONS)
COLUMNS = [
    "dataset",
    "method",
    "decomposition",
    "post_processing",
    "seeds",
    "n_concepts",
    "gini_sparsity_mean",
    "gini_sparsity_sd",
    "l2_overlap_mean",
    "l2_overlap_sd",
    "instability",
    "status",
]


def post_processing(spec: str) -> Callable[[np.ndarray], np.ndarray]:
    """The bank transform of a post-processing: "none", or "+"-joined steps
    applied in order, "lowdin" (symmetric orthogonalization) and
    "magnitude:<d>" (keep the fraction d of each concept's largest
    coordinates).

    Raises:
        ValueError: An unknown step.
    """
    steps = [] if spec == "none" else spec.split("+")

    def apply(W: np.ndarray) -> np.ndarray:
        for step in steps:
            name, _, value = step.partition(":")
            if name == "lowdin":
                W = lowdin_orthogonalize(W)
            elif name == "magnitude":
                W = magnitude_sparsify(W, math.ceil(float(value) * W.shape[1]))
            else:
                raise ValueError(f"unknown post-processing step {step!r}")
        return W

    return apply


def load_bank(
    run: Path, decomposition: str, all_components: bool
) -> Optional[np.ndarray]:
    """Concepts of a run's untrimmed bank (plus its negative bank with
    all_components); None if missing or empty."""
    folder = run / "concept" / decomposition
    found = sorted(
        folder.glob(f"combined_concept_{decomposition}_*_raw_dense.pth")
    ) or sorted(folder.glob(f"combined_concept_{decomposition}_*_raw.pth"))
    if not found:
        return None
    paths = found[:1]
    if all_components:
        paths += sorted(
            folder.glob(f"combined_negative_concept_{decomposition}_*_raw.pth")
        )[:1]
    parts = []
    for path in paths:
        bank = torch.load(path, map_location="cpu", weights_only=False)
        c = torch.as_tensor(bank["concepts"]).double().numpy()
        if c.ndim == 2 and len(c):
            parts.append(c)
    return np.concatenate(parts) if parts else None


def evaluate_row(row: Dict, posts: List[str] = POSTS) -> List[Dict]:
    """One output record per post-processing of a spec row."""
    seeds = []
    for seed, run in enumerate(row["runs"], start=1):
        W = load_bank(
            ROOT / run, row["decomposition"], row.get("all_components", False)
        )
        if W is not None:
            seeds.append((seed, W))
    records = []
    for post in posts:
        rec = {k: row[k] for k in ("dataset", "method", "decomposition")} | {
            "post_processing": post
        }
        if not seeds:
            records.append(rec | {"status": "no bank"})
            continue
        banks = [post_processing(post)(W) for _, W in seeds]
        rec |= {
            "seeds": ",".join(str(s) for s, _ in seeds),
            "n_concepts": "/".join(str(len(W)) for W in banks),
        }
        for key, metric in (
            ("gini_sparsity", dm.gini_sparsity),
            ("l2_overlap", dm.l2_overlap),
        ):
            values = [metric(W) for W in banks]
            rec[f"{key}_mean"] = float(np.mean(values))
            rec[f"{key}_sd"] = float(np.std(values))
        rec["instability"] = dm.instability(banks)
        rec["status"] = "ok" if len(banks) > 1 else "1 seed: no stability"
        records.append(rec)
    return records


def _summary(rec: Dict) -> str:
    """One progress line per record."""
    name = (
        f"{rec['dataset']} | {rec['method']} | {rec['decomposition']} | "
        f"{rec['post_processing']}"
    )
    if "gini_sparsity_mean" not in rec:
        return f"{name}: {rec['status']}"
    return (
        f"{name}: sparsity {rec['gini_sparsity_mean']:.2f}, "
        f"instability {rec['instability']:.3f}, "
        f"overlap {rec['l2_overlap_mean']:.2f} ({rec['status']})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in spec["rows"]:
            for rec in evaluate_row(row):
                writer.writerow(
                    {
                        k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in rec.items()
                    }
                )
                fh.flush()
                print(_summary(rec), flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
