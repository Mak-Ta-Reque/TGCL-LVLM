#!/usr/bin/env python3
"""Test-set evaluation of finished concept banks, without a new decomposition.

The explainer's answers and answer states on the test images do not depend on
the concept bank (scripts/rerank_explanations.py), so every bank of a
concept-quality spec (configs/dictionary_metrics/*.json; the rows of
tgcl/reporting/dictionary_metrics.table_rows, TGCL's SNMF once per trim of its
Lowdin combination) is evaluated on one source run's stored test states:

    explain  the source's explainer passes (pipeline step 6), one per setting,
             when it has none yet (GPU)
    build    one evaluation run per (row, seed): the bank (post-processed and
             re-grounded for TGCL's combination; the baselines' rejected
             directions included with all_components) and its re-ranked
             explanations per setting, plus Acc / macro F1 of the top concept's
             class on every test image (CPU)
    run      pipeline step 7 on every evaluation run and setting: concept
             insertion/deletion curves, CLIP/BERT of the retrieved concepts,
             word alignment (GPU)
    curves   only the concept insertion/deletion curves, for every run and
             setting without them (or all, --overwrite), one VLM load per GPU
    table    per setting, mean (sd) over seeds: Acc, F1, the concept
             insertion-deletion gap (net of its random-order control), CLIP
             and BERT of the rank-1 concept

Matching follows each method's pipeline: TGCL's runs are centred on the
explainer's calibration mean (EXPL_CENTER=calibration), the baselines' are not.

    S=configs/dictionary_metrics/gemma3n_coco10.json O=outputs/bank_eval/coco10
    SRC='artifacts/test_states/coco10/seed{seed}'
    python scripts/evaluate_banks.py build --spec $S --source "$SRC" --out $O
    python scripts/evaluate_banks.py run --out $O --dataset-preset coco
    python scripts/evaluate_banks.py table --out $O

Faithfulness (tgcl/evaluation/concept_insertion_deletion.py): the concepts,
ordered by their activation on each answer state, are deleted from it one by
one, or inserted back into its concept-free part; gap = area(insertion) -
area(reversed deletion), and the reported net gap subtracts the same gap in
random order.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
from statistics import pstdev
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from dictionary_metrics import post_processing
from rerank_explanations import rerank_file
from sparsify_concepts import reground
import torch

from tgcl.evaluation.concept_insertion_deletion import ConceptCurveEvaluator
from tgcl.evaluation.concept_insertion_deletion import read_summary
from tgcl.evaluation.concept_insertion_deletion import (
    SUMMARY as CONCEPT_CURVES_SUMMARY,
)
from tgcl.evaluation.concept_insertion_deletion import write_outputs
from tgcl.experiments import free_gpus
from tgcl.experiments import Job
from tgcl.experiments import pipeline_job
from tgcl.experiments import Scheduler
from tgcl.helpers.run_layout import eval_dir
from tgcl.helpers.run_layout import explanations_dir
from tgcl.reporting import dictionary_metrics as dm
from tgcl.reporting import intermediates as im
from tgcl.reporting.report_metrics import compute_metrics
from tgcl.reporting.report_metrics import load_predictions_from_json

ROOT = Path(__file__).resolve().parent.parent
# Evaluation setting (EVAL_TAG) -> its configs/paper preset ("" = the default
# multiple-choice question with the image).
SETTINGS = {
    "": None,
    "image_only": "eval_image_only",
    "text_only": "eval_text_only",
    "combined": "eval_combined",
}
SETTING_NAMES = {
    "": "MCQ",
    "image_only": "Image-only",
    "text_only": "Text-only",
    "combined": "Combined",
}
SOURCE_DECOMPOSITION = "snmf"  # the source run's explanations
RATIO_TAG = "cr0.6"  # bank file tag of the presets (common.env)
PER_CONCEPT = (
    "concept_names",
    "text_grounding",
    "image_grounding_paths",
    "image_grounding_bboxes",
    "image_grounding_masks",
    "image_grounding_predictions",
)
META = "bank_eval.json"
ACCURACY = "test_accuracy.json"
VLM = "google/gemma-3n-E4B-it"
LAYER = "model.language_model.norm"


# ------------------------------------------------------------------- build


def merged_bank(
    run: Path, decomposition: str, all_components: bool
) -> Optional[Dict[str, Any]]:
    """A run's untrimmed bank with its per-concept fields (plus its negative
    bank with all_components, as scripts/dictionary_metrics.load_bank); None if
    missing or empty."""
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
    merged: Dict[str, Any] = {"concepts": []}
    for path in paths:
        bank = torch.load(path, map_location="cpu", weights_only=False)
        c = torch.as_tensor(bank["concepts"]).float()
        if c.ndim != 2 or c.shape[0] == 0:
            continue
        merged["concepts"].append(c)
        for key in PER_CONCEPT:
            values = bank.get(key)
            if values is not None and len(values) == len(c):
                merged.setdefault(key, []).extend(list(values))
        for key in ("decomposition_method", "analysis_model", "selected_layer"):
            merged.setdefault(key, bank.get(key))
    if not merged["concepts"]:
        return None
    merged["concepts"] = torch.cat(merged["concepts"])
    n = len(merged["concepts"])
    for key in PER_CONCEPT:  # a field missing in one part is dropped
        if key in merged and len(merged[key]) != n:
            del merged[key]
    return merged


def processed_bank(bank: Dict[str, Any], post: str) -> Dict[str, Any]:
    """The bank after a post-processing (scripts/dictionary_metrics), its
    words re-grounded through the VLM's output head as the pipeline does."""
    if post == "none":
        return bank
    out = dict(bank)
    w = post_processing(post)(bank["concepts"].double().numpy())
    out["concepts"] = torch.tensor(w, dtype=torch.float32)
    out["post_processing"] = post
    reground(out, VLM, LAYER)
    return out


def run_name(method: str, decomposition: str, post: str, seed: int) -> str:
    """Directory name of an evaluation run."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in method.lower())
    slug = "_".join(p for p in slug.split("_") if p)
    steps = dict(s.partition(":")[::2] for s in post.split("+"))
    if post == "none":
        suffix = ""
    elif "magnitude" in steps:  # TGCL's combination: Lowdin + keep d
        suffix = ("" if "lowdin" in steps else "_nolowdin") + (
            "_keep" + steps["magnitude"]
        )
    else:
        suffix = "_" + "_".join(steps)  # e.g. Lowdin alone, every coordinate
    return f"{slug}_{decomposition}{suffix}_seed{seed}"


def test_accuracy(
    source_json: Path, bank: Dict[str, Any], centred: bool
) -> Dict[str, float]:
    """Acc / macro F1 of the rank-1 concept's class on every test image."""
    with tempfile.TemporaryDirectory() as tmp:  # TMPDIR
        out = Path(tmp) / "vlm_explanations.json"
        rerank_file(source_json, bank, out, 3, centred=centred)
        rows = load_predictions_from_json(out)
    return compute_metrics(rows, classes=sorted({r[0] for r in rows}))


def build_run(
    out_run: Path,
    bank: Dict[str, Any],
    meta: Dict[str, Any],
    source: Path,
    per_class: int,
) -> None:
    """Write one evaluation run: the bank, its re-ranked explanations per
    setting (first per_class test images per class; 0 = all), Acc / F1 on
    every test image, and the run's metadata."""
    decomposition, centred = meta["decomposition"], meta["centred"]
    bank_path = (
        out_run
        / "concept"
        / decomposition
        / f"combined_concept_{decomposition}_{RATIO_TAG}_raw.pth"
    )
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, bank_path)
    accuracy = {}
    for tag in SETTINGS:
        src = (
            explanations_dir(source, tag)
            / SOURCE_DECOMPOSITION
            / "vlm_explanations.json"
        )
        if not src.exists():
            continue
        dst = explanations_dir(out_run, tag) / decomposition
        if dst.exists():
            shutil.rmtree(dst)
        rerank_file(
            src,
            bank,
            dst / "vlm_explanations.json",
            3,
            centred=centred,
            per_class=per_class,
        )
        accuracy[tag] = test_accuracy(src, bank, centred)
    (out_run / ACCURACY).write_text(json.dumps(accuracy, indent=1))
    (out_run / META).write_text(json.dumps(meta, indent=1))


def cmd_build(args) -> None:
    spec = json.loads(args.spec.read_text())
    for row in spec["rows"]:
        method, decomposition = row["method"], row["decomposition"]
        for seed, run in enumerate(row["runs"], start=1):
            bank = merged_bank(
                ROOT / run, decomposition, row.get("all_components", False)
            )
            if bank is None:
                print(f"skip {method} {decomposition} seed {seed}: no bank")
                continue
            source = ROOT / args.source.format(seed=seed)
            posts = dm.posts_of(method, decomposition)
            if args.tgcl_snmf_posts and posts != ["none"]:
                posts = args.tgcl_snmf_posts.split(",")
            for post in posts:
                out_run = args.out / run_name(method, decomposition, post, seed)
                if (out_run / META).exists() and not args.overwrite:
                    continue
                meta = {
                    "dataset": row["dataset"],
                    "method": method,
                    "decomposition": decomposition,
                    "post_processing": post,
                    "seed": seed,
                    "bank_run": run,
                    "source": str(source.relative_to(ROOT)),
                    "centred": dm.is_tgcl(method),
                    "per_class": args.per_class,
                    "n_concepts": int(len(bank["concepts"])),
                }
                build_run(
                    out_run,
                    processed_bank(bank, post),
                    meta,
                    source,
                    args.per_class,
                )
                print(f"built {out_run.name}", flush=True)


# --------------------------------------------------------------------- run


def presets(dataset_preset: str, tag: str) -> List[str]:
    """configs/paper presets of a setting (step 6/7 read only the dataset and
    evaluation settings; tgcl_sam fixes the bank file tag and centring)."""
    extra = [SETTINGS[tag]] if SETTINGS[tag] else []
    return ["common", dataset_preset, "tgcl_sam", *extra]


def evaluated(run: Path, decomposition: str, tag: str) -> bool:
    """Whether step 7 has finished for a setting (CLIP/BERT and the concept
    insertion/deletion curves)."""
    folder = eval_dir(run, tag) / decomposition
    return (folder / "clip_bert_topk_table.csv").exists() and (
        folder / CONCEPT_CURVES_SUMMARY
    ).exists()


def evaluation_jobs(out: Path, dataset_preset: str) -> List[Job]:
    """One step-7 job per evaluation run and setting still to do."""
    jobs = []
    for meta_path in sorted(out.glob(f"*/{META}")):
        run = meta_path.parent
        meta = json.loads(meta_path.read_text())
        for tag in SETTINGS:
            src = explanations_dir(run, tag) / meta["decomposition"]
            if not src.exists() or evaluated(run, meta["decomposition"], tag):
                continue
            job = pipeline_job(
                f"{run.name}_{tag or 'mcq'}",
                run,
                presets(dataset_preset, tag),
                ["--skip-to-step", "7", "--stop-after-step", "7"],
            )
            job.env.update(
                {
                    "DECOMP_METHODS": meta["decomposition"],
                    "CONCEPT_SPARSIFY": "",
                    "SEED": str(meta["seed"]),
                }
            )
            jobs.append(job)
    return jobs


def explain_jobs(
    source: Path,
    bank_run: Path,
    dataset_preset: str,
    seed: int = 1,
    needs: Optional[set] = None,
) -> List[Job]:
    """The source's explainer passes (step 6), one per setting, on a finished
    TGCL run's bank (needs: jobs that must have finished first)."""
    jobs = []
    for tag in SETTINGS:
        if (explanations_dir(source, tag) / SOURCE_DECOMPOSITION).exists():
            continue
        job = pipeline_job(
            f"explain_{source.name}_{tag or 'mcq'}",
            source,
            presets(dataset_preset, tag),
            ["--skip-to-step", "6", "--stop-after-step", "6"],
            reuse_run=bank_run,
            evaluation=True,
            needs=set(needs or ()),
        )
        job.env.update(
            {
                "DECOMP_METHODS": SOURCE_DECOMPOSITION,
                "CONCEPT_SPARSIFY": "",
                "SEED": str(seed),
            }
        )
        jobs.append(job)
    return jobs


def schedule(jobs: List[Job], logs: Path, args) -> None:
    if not jobs:
        print("nothing to do")
        return
    ok = Scheduler(
        jobs,
        logs=logs,
        gpus=set(map(int, args.gpus.split(","))) if args.gpus else None,
        max_gpus=args.max_gpus,
        free_mib=2000,
        max_cpu_jobs=2,
        max_eval_jobs=args.max_gpus,
        max_attempts=2,
        retry_s=120,
        poll_s=15,
        threads_per_job=8,
    ).run()
    raise SystemExit(0 if ok else 1)


def cmd_run(args) -> None:
    schedule(
        evaluation_jobs(args.out, args.dataset_preset), args.out / "logs", args
    )


def curve_items(out: Path, overwrite: bool = False) -> List[tuple]:
    """(run, setting, decomposition, seed) of every evaluation run and
    setting whose concept curves are missing (all with overwrite)."""
    items = []
    for meta_path in sorted(out.glob(f"*/{META}")):
        run = meta_path.parent
        meta = json.loads(meta_path.read_text())
        decomposition = meta["decomposition"]
        for tag in SETTINGS:
            explanations = explanations_dir(run, tag) / decomposition
            if not (explanations / "vlm_explanations.json").exists():
                continue
            done = read_summary(eval_dir(run, tag) / decomposition)
            if done is None or overwrite:
                items.append((run, tag, decomposition, meta["seed"]))
    return items


def cmd_curves(args) -> None:
    """Concept curves of every pending (run, setting): with --shard k/n this
    process (one GPU) takes every n-th item from k; otherwise it starts one
    such worker per free GPU and waits."""
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        items = curve_items(args.out, args.overwrite)[k::n]
        if not items:
            return
        evaluator = ConceptCurveEvaluator(VLM, LAYER)
        for run, tag, decomposition, seed in items:
            explanations = explanations_dir(run, tag) / decomposition
            bank = (
                run
                / "concept"
                / decomposition
                / f"combined_concept_{decomposition}_{RATIO_TAG}_raw.pth"
            )
            result = evaluator.evaluate(
                explanations / "vlm_explanations.json", bank, seed
            )
            write_outputs(result, eval_dir(run, tag) / decomposition)
            s = result["summary"]
            print(
                f"{run.name} {tag or 'mcq'}: gap {s['gap']:.3f} random "
                f"{s['gap_random']:.3f} net {s['gap_net']:.3f}",
                flush=True,
            )
        return
    todo = curve_items(args.out, args.overwrite)
    gpus = (
        [int(g) for g in args.gpus.split(",")]
        if args.gpus
        else free_gpus(None, 2000, set())
    )[: args.max_gpus]
    if not todo or not gpus:
        print(f"{len(todo)} pending, {len(gpus)} GPU(s): nothing started")
        return
    print(f"{len(todo)} (run, setting) pairs on GPUs {gpus}", flush=True)
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    workers = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, __file__, "curves", "--out", str(args.out),
               "--shard", f"{i}/{len(gpus)}"]  # fmt: skip
        if args.overwrite:
            cmd.append("--overwrite")
        log = open(logs / f"curves_{i}.log", "w", encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        workers.append(
            (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                              env=env), log)  # fmt: skip
        )
    failed = 0
    for proc, log in workers:
        failed += proc.wait() != 0
        log.close()
    left = len(curve_items(args.out))
    print(f"done: {failed} worker(s) failed, {left} pair(s) still pending")
    raise SystemExit(1 if failed or left else 0)


def cmd_explain(args) -> None:
    schedule(
        explain_jobs(args.source, args.bank_run, args.dataset_preset),
        args.source / "scheduler_logs",
        args,
    )


# ------------------------------------------------------------------- table

FIELDS = [
    "acc",
    "f1",
    "gap_net",
    "gap",
    "gap_random",
    "insertion_auc",
    "deletion_auc",
    "clip",
    "bert",
    "clip_random",
    "bert_random",
]
SHOWN = [("Acc↑", "acc"), ("F1↑", "f1"), ("Gap↑", "gap_net"),
         ("CLIP↑", "clip"), ("BERT↑", "bert")]  # fmt: skip


def run_scores(run: Path, meta: Dict[str, Any], tag: str) -> Dict[str, float]:
    """One evaluation run's scores in one setting (NaN where missing)."""
    decomposition = meta["decomposition"]
    nan = float("nan")
    acc = json.loads((run / ACCURACY).read_text()).get(tag) or {}
    im.export_run(run, [decomposition], [tag])
    curves = read_summary(eval_dir(run, tag) / decomposition) or {}
    rank1 = im.read_retrieval(run, decomposition, tag).get(1, {})
    return {
        "acc": acc.get("top1_accuracy", nan),
        "f1": acc.get("top1_f1_macro", nan),
        **{
            k: curves.get(k, nan)
            for k in (
                "gap_net",
                "gap",
                "gap_random",
                "insertion_auc",
                "deletion_auc",
            )
        },
        "clip": rank1.get("clip", nan),
        "bert": rank1.get("bert", nan),
        "clip_random": rank1.get("random_clip", nan),
        "bert_random": rank1.get("random_bert", nan),
    }


def _mean_sd(values: List[float]):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    return mean(values), pstdev(values)


def summarize(out: Path) -> List[Dict[str, Any]]:
    """Mean and sd over seeds per (method, decomposition, post, setting)."""
    groups: Dict[tuple, List[Dict[str, float]]] = {}
    for meta_path in sorted(out.glob(f"*/{META}")):
        meta = json.loads(meta_path.read_text())
        key = (
            meta["dataset"],
            meta["method"],
            meta["decomposition"],
            meta["post_processing"],
        )
        for tag in SETTINGS:
            scores = run_scores(meta_path.parent, meta, tag)
            groups.setdefault(key + (tag,), []).append(scores)
    records = []
    for (dataset, method, decomposition, post, tag), runs in groups.items():
        rec = {
            "dataset": dataset,
            "method": method,
            "decomposition": decomposition,
            "post_processing": post,
            "setting": SETTING_NAMES[tag],
            "seeds": len(runs),
        }
        for field in FIELDS:
            m, sd = _mean_sd([r[field] for r in runs])
            rec[f"{field}_mean"], rec[f"{field}_sd"] = m, sd
        records.append(rec)
    return records


def markdown(records: List[Dict[str, Any]], setting: str) -> str:
    """The table of one setting, rows in the paper's order."""
    by_key = {
        (r["method"], r["decomposition"], r["post_processing"]): r
        for r in records
        if r["setting"] == setting
    }
    lines = [
        "| Method | Decomp. | " + " | ".join(n for n, _ in SHOWN) + " |",
        "|" + "---|" * (2 + len(SHOWN)),
    ]
    for method, key, label, post in dm.table_rows():
        rec = by_key.get((method, key, post), {})
        cells = []
        for _, field in SHOWN:
            value = rec.get(f"{field}_mean", float("nan"))
            cells.append("–" if math.isnan(value) else f"{value:.3f}")
        lines.append(f"| {method} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_table(args) -> None:
    records = summarize(args.out)
    path = args.out / "results.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    k: (f"{v:.4f}" if isinstance(v, float) else v)
                    for k, v in rec.items()
                }
            )
    for setting in SETTING_NAMES.values():
        print(f"\n### {setting}\n\n{markdown(records, setting)}")
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--spec", type=Path, required=True)
    b.add_argument(
        "--source",
        required=True,
        help="source run with the test explanations ({seed} is replaced)",
    )
    b.add_argument("--out", type=Path, required=True)
    b.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="step-7 test images per class (default: all; Acc/F1 use all)",
    )
    b.add_argument("--overwrite", action="store_true")
    b.add_argument(
        "--tgcl-snmf-posts",
        default="",
        help="comma-separated post-processings for TGCL's SNMF rows instead "
        "of its combinations (e.g. none,lowdin: no trim); other rows as "
        "learned",
    )
    for name in ("run", "explain"):
        p = sub.add_parser(name)
        p.add_argument("--dataset-preset", required=True)
        p.add_argument("--gpus", default="")
        p.add_argument("--max-gpus", type=int, default=6)
        if name == "run":
            p.add_argument("--out", type=Path, required=True)
        else:
            p.add_argument("--source", type=Path, required=True)
            p.add_argument("--bank-run", type=Path, required=True)
    c = sub.add_parser("curves")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--gpus", default="")
    c.add_argument("--max-gpus", type=int, default=6)
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--shard", default="", help=argparse.SUPPRESS)
    t = sub.add_parser("table")
    t.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        args.out.mkdir(parents=True, exist_ok=True)
    {
        "build": cmd_build,
        "run": cmd_run,
        "explain": cmd_explain,
        "curves": cmd_curves,
        "table": cmd_table,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
