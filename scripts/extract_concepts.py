#!/usr/bin/env python3
"""Extract every concept bank of the paper tables from the beginning, in the
layout of the released concepts (Hugging Face abka03/TGCL-LVLM), so the
evaluation commands of the README run unchanged on the result.

For one dataset and every seed:

    concepts   pipeline steps 1-5 of TGCL (SAM segments, sliding windows),
               CoX-LMM, CRAFT and MCD, each decomposed with semi-NMF, SAE and
               SIMPLE. The baselines learn from the same seed's TGCL training
               images (MCD also from its SAM segments), pooled over all classes
               with as many concepts as there are classes.
               -> <out>/concepts/<dataset>/<method>_seed<s>/concept/<d>/
    test states  pipeline step 6 of TGCL (SAM) in the four evaluation settings:
               the explainer's answers and hidden states on the test images,
               which every concept bank is evaluated on (COCO-10: per seed;
               ImageNet-1k: seed 1, its test list is fixed)
               -> <out>/test_states/<dataset>[/seed<s>]/explanations*/snmf/

    python scripts/extract_concepts.py --dataset coco10 --dry-run
    python scripts/extract_concepts.py --dataset coco10 [--gpus 0,1,2,3]

Jobs run on the free GPUs (tgcl/experiments.py); re-running resumes, finished
steps are skipped.
"""

import argparse
from pathlib import Path
from typing import List

from evaluate_banks import explain_jobs

from tgcl.experiments import Job
from tgcl.experiments import pipeline_job
from tgcl.experiments import ROOT
from tgcl.experiments import Scheduler

DATASETS = {
    # dataset: (configs/paper preset, pooled concepts of the baselines)
    "coco10": ("coco", "10"),
    "imagenet1k": ("imagenet1k_quick", "1000"),
}
TGCL = ("tgcl_sam", "tgcl_sliding_window")
BASELINES = ("coxlmm", "craft", "mcd")
DECOMPOSITIONS = "snmf,simple,sae2"
SEEDS = (1, 2, 3)


def concept_jobs(dataset: str, out: Path, seeds=SEEDS) -> List[Job]:
    """Steps 1-5 of every method and seed."""
    preset, n_global = DATASETS[dataset]
    root = out / "concepts" / dataset
    jobs = []
    for seed in seeds:
        sam = f"{dataset}_tgcl_sam_s{seed}"
        for method in TGCL + BASELINES:
            run = root / f"{method}_seed{seed}"
            kw = {}
            if method in BASELINES:
                # the seed's TGCL training images (step-1 files); MCD also
                # its SAM segments (crops.json)
                kw = dict(
                    copy_from=root / f"tgcl_sam_seed{seed}",
                    copy_to=run,
                    copy_crops=method == "mcd",
                    needs={sam},
                )
            job = pipeline_job(
                f"{dataset}_{method}_s{seed}",
                run,
                ["common", preset, method],
                ["--stop-after-step", "5"],
                **kw,
            )
            job.env.update(
                {"SEED": str(seed), "DECOMP_METHODS": DECOMPOSITIONS}
            )
            if method in BASELINES:
                job.env["DECOMP_COMPONENTS_GLOBAL"] = n_global
            jobs.append(job)
    return jobs


def state_jobs(dataset: str, out: Path, seeds=SEEDS) -> List[Job]:
    """Step 6 of TGCL (SAM) in the four settings: the test states."""
    preset, _ = DATASETS[dataset]
    root = out / "concepts" / dataset
    jobs = []
    for seed in seeds if dataset == "coco10" else seeds[:1]:
        source = out / "test_states" / dataset
        if dataset == "coco10":
            source = source / f"seed{seed}"
        jobs += explain_jobs(
            source,
            root / f"tgcl_sam_seed{seed}",
            preset,
            seed=seed,
            needs={f"{dataset}_tgcl_sam_s{seed}"},
        )
    for job in jobs:  # unique names across seeds and datasets
        job.name = f"{dataset}_{job.name}"
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = concept_jobs(args.dataset, args.out) + state_jobs(
        args.dataset, args.out
    )
    if args.dry_run:
        for job in jobs:
            print(f"{job.name}: needs {sorted(job.needs) or '-'}")
        print(f"{len(jobs)} jobs -> {args.out}")
        return
    ok = Scheduler(
        jobs,
        logs=args.out / "logs" / args.dataset,
        gpus=set(map(int, args.gpus.split(","))) if args.gpus else None,
        max_gpus=args.max_gpus,
        free_mib=2000,
        max_cpu_jobs=2,
        max_eval_jobs=2,
        max_attempts=2,
        retry_s=300,
        poll_s=15,
        threads_per_job=8,
    ).run()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
