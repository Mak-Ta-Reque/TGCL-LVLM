#!/usr/bin/env python3
"""Paper tables from finished pipeline runs (CPU only, no model).

A configuration is a group of run directories that differ only in the seed:
directories named "<config>_seed<N>" are grouped under "<config>"; a
directory without that suffix is a one-seed group. Run layout and evaluation
tags: src/helpers/run_layout.py.

`tables` writes, for the given runs and decomposition methods:
  concept_quality.csv  Hoyer sparsity, near-zero fraction, overlap, instability
  token_hit_rate.csv   share of samples whose answer contains the concept word
  retrieval.csv        CLIP/BERT of retrieved concepts: Random, Text-only,
  Image-only, Combined
  faithfulness.csv     concept insertion/deletion: the gap (area(insertion) -
                       area(reversed deletion), concepts in activation order),
                       its random-order control, the net gap and both AUCs
  object_f1.csv       top-1/top-2 accuracy and macro F1 of concept identity vs
  the image class

All metrics are computed from editable intermediate files in
<run>/intermediate/ (formats: scripts/intermediates.py and the README written
there). `export` creates them from the pipeline outputs; `tables`
exports missing ones first but never overwrites, so edits are kept.

Usage:
    python scripts/paper_report.py tables \\
        --runs outputs/paper/tgcl_coco_seed1 --methods snmf,simple,sae2,pca \\
        --out-dir outputs/paper/_report/tgcl_coco
"""

import argparse
from collections import defaultdict
import csv
from itertools import combinations
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import numpy as np

from tgcl.helpers.run_layout import explanations_json
from tgcl.helpers.run_layout import validate_eval_tag
import tgcl.reporting.intermediates as im
from tgcl.reporting.report_metrics import compute_metrics
from tgcl.reporting.report_metrics import concept_overlap
from tgcl.reporting.report_metrics import effective_coordinates
from tgcl.reporting.report_metrics import effective_rank_fraction
from tgcl.reporting.report_metrics import gini_index
from tgcl.reporting.report_metrics import hoyer_selectivity
from tgcl.reporting.report_metrics import matched_cosine_similarity
from tgcl.reporting.report_metrics import near_zero_fraction
from tgcl.reporting.report_metrics import NEAR_ZERO_THRESHOLD
from tgcl.reporting.report_metrics import score_correlation

# Retrieval columns: (column, EVAL_TAG of the pipeline run that produced it).
RETRIEVAL_VARIANTS = (
    ("Text-only", "text_only"),
    ("Image-only", "image_only"),
    ("Combined", "combined"),
)
# The random-concept baseline is scored against the image-only answers.
RANDOM_BASELINE_TAG = "image_only"
RANKS = (1, 2, 3)
_SEED_SUFFIX = re.compile(r"_seed\d+$")


# ---------------------------------------------------------------- grouping


def group_runs(runs: Iterable[Path]) -> Dict[str, List[Path]]:
    """{config name: [run dirs]} -- runs differing only in "_seed<N>" share a
    group."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for run in runs:
        run = Path(run)
        groups[_SEED_SUFFIX.sub("", run.name)].append(run)
    return {name: sorted(dirs) for name, dirs in sorted(groups.items())}


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _std(values: List[float]) -> Optional[float]:
    return float(np.std(values)) if len(values) > 1 else None


# ---------------------------------------------------------------- metrics


MAX_SCORE_TOKENS = (
    20000  # answer tokens used for the score correlation (evenly spaced)
)


def answer_states(run: Path, method: str) -> Optional[np.ndarray]:
    """The run's answer-token hidden states (base explanations), at most
    MAX_SCORE_TOKENS evenly spaced."""
    acts = explanations_json(run, method).with_name(
        "vlm_explanations_activations.npy"
    )
    if not acts.exists():
        return None
    a = np.load(acts, mmap_mode="r")
    step = max(1, len(a) // MAX_SCORE_TOKENS)
    return np.asarray(a[::step][:MAX_SCORE_TOKENS], dtype=np.float64)


def concept_quality(
    runs: List[Path],
    method: str,
    near_zero_threshold: float = NEAR_ZERO_THRESHOLD,
) -> dict:
    """Dictionary metrics of one configuration (mean over seeds; instability
    over seed pairs)."""
    pairs = [
        (run, c)
        for run in runs
        for c in [im.read_concepts(run, method)]
        if c is not None
    ]
    banks = [c for _, c in pairs]
    hoyer = [hoyer_selectivity(c) for c in banks]
    gini = [gini_index(c) for c in banks]
    n_eff = [effective_coordinates(c) for c in banks]
    erank = [effective_rank_fraction(c) for c in banks if c.shape[0] > 1]
    score_corr = [
        x
        for run, c in pairs
        for x in [score_correlation(c, answer_states(run, method))]
        if x is not None
    ]
    near_zero = [near_zero_fraction(c, near_zero_threshold) for c in banks]
    overlap = [
        concept_overlap(c, near_zero_threshold) for c in banks if c.shape[0] > 1
    ]  # undefined for 1 concept
    # float64: instability is 1 - cos with cos close to 1, below float32
    # resolution.
    instability = [
        1.0
        - matched_cosine_similarity(
            a.double().numpy(),
            b.double().numpy(),
            sign_invariant=method == "pca",  # PCA signs are arbitrary
        )
        for a, b in combinations(banks, 2)
    ]
    return {
        "n_seeds": len(banks),
        "n_concepts": int(banks[0].shape[0]) if banks else None,
        "sparsity_hoyer": _mean(hoyer),
        "sparsity_hoyer_std": _std(hoyer),
        "near_zero_frac": _mean(near_zero),
        "near_zero_frac_std": _std(near_zero),
        "near_zero_threshold": near_zero_threshold,
        "gini": _mean(gini),
        "effective_coordinates": _mean(n_eff),
        "effective_rank_frac": _mean(erank),
        "score_correlation": _mean(score_corr),
        "overlap": _mean(overlap),
        "overlap_std": _std(overlap),
        "instability": _mean(instability),
    }


def token_hit_rate(run: Path) -> List[dict]:
    """Per concept: samples whose answer contained the concept word.

    From intermediate/token_hits.csv; runs without a token-of-interest mask
    (e.g. the last-token hook) yield [].
    """
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for concept, found in im.read_token_hits(run):
        counts[concept][0] += 1
        counts[concept][1] += int(found)
    rows = [
        {
            "concept": c,
            "n_samples": n,
            "n_found": k,
            "hit_rate": k / n if n else None,
        }
        for c, (n, k) in sorted(counts.items())
    ]
    if rows:
        n_all = sum(r["n_samples"] for r in rows)
        k_all = sum(r["n_found"] for r in rows)
        rows.append(
            {
                "concept": "ALL",
                "n_samples": n_all,
                "n_found": k_all,
                "hit_rate": k_all / n_all if n_all else None,
            }
        )
    return rows


WORD_ALIGNMENT_METRICS = (
    "word_clipscore",
    "random_word_clipscore",
    "word_bertscore",
    "random_word_bertscore",
    "phrase_bertscore",
    "random_phrase_bertscore",
)


def word_alignment_table(
    runs: List[Path], method: str, eval_tags: Iterable[str] = ("", "grid")
) -> List[dict]:
    """Concept-word alignment (eval/word_alignment_eval.py) per evaluation tag
    and k, mean over seeds."""
    rows = []
    for tag in eval_tags:
        tables = [im.read_word_alignment(run, method, tag) for run in runs]
        ks = sorted({k for t in tables for m in t.values() for k in m})
        for k in ks:
            row = {"eval_tag": tag, "rank_k": k}
            for metric in WORD_ALIGNMENT_METRICS:
                row[metric] = _mean(
                    [
                        t[metric][k]
                        for t in tables
                        if t.get(metric, {}).get(k) is not None
                    ]
                )
            if any(row[m] is not None for m in WORD_ALIGNMENT_METRICS):
                rows.append(row)
    return rows


def retrieval_table(runs: List[Path], method: str) -> List[dict]:
    """Rows per (rank, metric) with the Random / Text-only / Image-only /
    Combined columns."""
    tables = {
        tag: [im.read_retrieval(run, method, tag) for run in runs]
        for tag in {tag for _, tag in RETRIEVAL_VARIANTS}
        | {RANDOM_BASELINE_TAG}
    }

    def column(tag: str, rank: int, key: str) -> Optional[float]:
        return _mean(
            [t[rank][key] for t in tables[tag] if key in t.get(rank, {})]
        )

    rows = []
    for rank in RANKS:
        for metric in ("bert", "clip"):
            row = {
                "rank_k": rank,
                "metric": metric,
                "Random": column(RANDOM_BASELINE_TAG, rank, f"random_{metric}"),
            }
            for name, tag in RETRIEVAL_VARIANTS:
                row[name] = column(tag, rank, metric)
            if any(
                v is not None
                for k, v in row.items()
                if k not in ("rank_k", "metric")
            ):
                rows.append(row)
    return rows


FAITHFULNESS_KEYS = (
    "gap",
    "gap_random",
    "gap_net",
    "insertion_auc",
    "deletion_auc",
)


def faithfulness(
    runs: List[Path], method: str, eval_tag: str = ""
) -> Optional[dict]:
    """Concept insertion/deletion (tgcl/evaluation/concept_insertion_deletion
    .py) of one configuration, mean (and sd) over seeds; None without curves.

    gap = area(insertion) - area(reversed deletion), concepts in activation
    order; gap_random the same in random order; gap_net = gap - gap_random.
    """
    per_seed = [
        s
        for run in runs
        if (s := im.read_concept_curves(run, method, eval_tag)) is not None
    ]
    if not per_seed:
        return None
    out = {
        "n_seeds": len(per_seed),
        "n_images": int(np.mean([s["n_images"] for s in per_seed])),
        "n_concepts": int(np.mean([s["n_concepts"] for s in per_seed])),
    }
    for key in FAITHFULNESS_KEYS:
        values = [s[key] for s in per_seed if s.get(key) is not None]
        out[key] = _mean(values)
        out[f"{key}_std"] = _std(values)
    return out


def object_f1(
    runs: List[Path],
    method: str,
    eval_tag: str = "",
    classes: Optional[List[str]] = None,
) -> Optional[dict]:
    """Concept identity vs image class (mean over seeds); None without
    explanations.

    Single-object evaluations only: the class is the image's parent folder.
    """
    per_seed = []
    for run in runs:
        rows = im.read_object_predictions(run, method, eval_tag)
        if rows:
            run_classes = classes or sorted({r[0] for r in rows})
            per_seed.append(compute_metrics(rows, classes=run_classes))
    if not per_seed:
        return None
    keys = ("top1_accuracy", "top1_f1_macro", "top2_accuracy", "top2_f1_macro")
    out = {
        "n_seeds": len(per_seed),
        "n_images": int(np.mean([m["n_images"] for m in per_seed])),
    }
    out.update({k: _mean([m[k] for m in per_seed]) for k in keys})
    return out


# ---------------------------------------------------------------- output


def write_csv(rows: List[dict], path: Path) -> None:
    """Write rows; keys starting with "_" (in-memory extras) are left out."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        columns.extend(
            k for k in row if k not in columns and not k.startswith("_")
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _fmt(value: Optional[float], digits: int) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def markdown_table(rows: List[dict], columns: List[tuple]) -> str:
    """columns: (key, header, digits or None for text, "max"/"min"/None for bold
    best)."""
    best = {}
    for key, _, _, direction in columns:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if direction and values:
            best[key] = max(values) if direction == "max" else min(values)
    lines = [
        "| " + " | ".join(h for _, h, _, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for r in rows:
        cells = []
        for key, _, digits, _ in columns:
            value = r.get(key)
            if digits is None:
                cells.append(str(value))
                continue
            text = _fmt(value, digits)
            cells.append(
                f"**{text}**" if key in best and value == best[key] else text
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- commands


def cmd_tables(args) -> None:
    """The `tables` command: every paper table of the given runs (seeds pooled
    per configuration) into --out_dir."""
    out_dir = Path(args.out_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    classes = (
        [c.strip() for c in args.classes.split(",")] if args.classes else None
    )
    faith_tags = [
        validate_eval_tag(t) for t in args.faithfulness_tags.split(",")
    ]
    objf1_tag = validate_eval_tag(args.objf1_tag)
    export_intermediates(args.runs, methods, faith_tags + [objf1_tag])
    groups = group_runs(args.runs)

    quality, hits, retrieval, alignment, faith, objf1 = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for group, runs in groups.items():
        for run in runs:
            hits += [
                {"config": group, "run": run.name, **r}
                for r in token_hit_rate(run)
            ]
        for method in methods:
            ident = {"config": group, "method": method}
            quality.append(
                {
                    **ident,
                    **concept_quality(runs, method, args.near_zero_threshold),
                }
            )
            retrieval += [{**ident, **r} for r in retrieval_table(runs, method)]
            alignment += [
                {**ident, **r} for r in word_alignment_table(runs, method)
            ]
            for tag in faith_tags:
                row = faithfulness(runs, method, tag)
                if row:
                    faith.append({**ident, "eval_tag": tag, **row})
            metrics = object_f1(runs, method, objf1_tag, classes)
            if metrics:
                objf1.append({**ident, "eval_tag": objf1_tag, **metrics})

    write_csv(quality, out_dir / "concept_quality.csv")
    write_csv(hits, out_dir / "token_hit_rate.csv")
    write_csv(retrieval, out_dir / "retrieval.csv")
    write_csv(alignment, out_dir / "word_alignment.csv")
    write_csv(faith, out_dir / "faithfulness.csv")
    write_csv(objf1, out_dir / "object_f1.csv")


def export_intermediates(
    runs: Iterable[Path],
    methods: List[str],
    eval_tags: List[str],
    overwrite: bool = False,
) -> None:
    """Fill in <run>/intermediate/; existing files are kept unless overwrite."""
    for run in runs:
        written = im.export_run(
            Path(run), methods, eval_tags, overwrite=overwrite
        )
        if written:
            print(
                f"{Path(run).name}: wrote {len(written)} intermediate file(s)"
            )


def cmd_export(args) -> None:
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tags = [validate_eval_tag(t) for t in args.eval_tags.split(",")]
    export_intermediates(args.runs, methods, tags, overwrite=args.overwrite)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser(
        "tables", help="per-configuration tables for the given runs"
    )
    t.add_argument("--runs", nargs="+", type=Path, required=True)
    t.add_argument("--methods", default="snmf")
    t.add_argument("--out-dir", required=True)
    t.add_argument(
        "--classes",
        default="",
        help="comma-separated classes for Obj-F1 (default: from image folders)",
    )
    t.add_argument(
        "--faithfulness-tags",
        default=",grid",
        help="comma-separated EVAL_TAGs for faithfulness ('' = untagged)",
    )
    t.add_argument(
        "--objf1-tag",
        default="",
        help="EVAL_TAG of the single-object evaluation used for Obj-F1 (class "
        "= image folder)",
    )
    t.add_argument(
        "--near-zero-threshold", type=float, default=NEAR_ZERO_THRESHOLD
    )
    t.set_defaults(func=cmd_tables)

    e = sub.add_parser(
        "export",
        help="write editable intermediate files (input of all metrics)",
    )
    e.add_argument("--runs", nargs="+", type=Path, required=True)
    e.add_argument("--methods", default="snmf")
    e.add_argument(
        "--eval-tags",
        default=",grid",
        help="comma-separated EVAL_TAGs ('' = untagged)",
    )
    e.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate files that already exist",
    )
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
