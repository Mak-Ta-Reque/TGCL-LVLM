#!/usr/bin/env python3
"""Run the paper experiments on the free GPUs of this machine.

    python scripts/run_experiments.py DATASET [--dry-run]

    DATASET          jobs
    coco             TGCL (SAM, sliding window), CoX-LMM, CRAFT, MCD
    coco_full        the coco runs on uncropped test images
    coco_raw_crops   TGCL with segments passed unresized
    imagenet5        all methods on 5 ImageNet classes (JSON subset)
    imagenet5_quick  the same with 20 images per class: minutes
    imagenet1k       TGCL on ImageNet-1k, class-sharded (--shards 8)

    # 1000 classes with Qwen2.5-VL, reusing the SAM segments of the Gemma run:
    python scripts/run_experiments.py tgcl --dataset imagenet1k_quick \\
        --model qwen2_5_vl_7b --no-variants --reuse-segments-from outputs/paper

Every job is one call of scripts/run_full_pipeline.py with the presets of
configs/paper/
(or a small CPU step). A job starts when its inputs exist and a GPU is free:
below
--free-mib used and not assigned to any running pipeline process (DEVICE of
run_full_pipeline.py, so a run between two steps keeps its GPU). Failed jobs
are retried; at most --max-eval-jobs explainer evaluations run at
once (each holds its results in host RAM). Re-running the command resumes: the
pipeline skips the steps whose outputs exist. Logs:
<out>/_logs/<experiment>/<job>.log.

coco: the baselines reuse the TGCL step-1 image lists and step-2 crops (MCD the
SAM
masks, CRAFT the sliding windows), so all methods see the same images and
segments.
imagenet1k: TGCL per crop mode as --shards class shards (CONCEPT_SHARD),
sliding-window
crops on the CPU, then scripts/merge_concept_shards.py and the evaluations on
the merged run.
"""

import argparse
import contextlib
from dataclasses import dataclass
from dataclasses import field
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
EVALS = ["eval_grid", "eval_text_only", "eval_image_only", "eval_combined"]
STEP1_FILES = ("objects.csv", "concepts_to_images.json")
# --smoke: every stage on a few images (minutes), to check a setup before the
# full run.
# No purity filter: with 5 images per class a concept rarely reaches the 0.6 of
# the paper runs;
# 4 pooled concepts: CoX-LMM keeps only the ~10 smoke images whose answer names
# a class.
# 10 images per class: with 5, Qwen2.5-VL left every SAE latent without a sample
# (empty bank).
SMOKE_ENV = {
    "IMAGE_BUDGET": "80",
    "NUM_CONCEPT": "6",
    "MAX_IMAGES_PER_TAG": "10",
    "BAG_SIZE": "50",
    "EXPL_MAX_IMAGES": "6",
    "CLEAN_EXAMPLE_RATIO": "0",
    "DECOMP_COMPONENTS_GLOBAL": "4",
}
# The SAE needs more samples than a smoke run has: its latents die (empty bank)
# at random.
SMOKE_SKIP_METHODS = {"sae2"}
# A run's settings come from configs/paper only. .env holds machine-specific
# values: only these keys (and
# PYTHON_<MODEL>, a model's interpreter) are passed on, and
# the pipeline does not load
# .env itself (XLVLMS_DOTENV=0): its experiment settings (VLM_MODEL, PROMPT,
# NUM_CONCEPT, ...) filled every
# variable a preset leaves unset, so a run from the main checkout differed from
# the same run in a worktree.
DOTENV_KEYS = (
    "HF_TOKEN",
    "HF_HOME",
    "TORCH_HOME",
    "NLTK_DATA",
    "PIP_CACHE_DIR",
    "MPLCONFIGDIR",
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TMPDIR",
)


def dotenv_path() -> Optional[Path]:
    """This checkout's .env, else the main checkout's (a git worktree has none
    of its own)."""
    if (ROOT / ".env").exists():
        return ROOT / ".env"
    try:
        common = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    env_file = Path(common).parent / ".env"
    return env_file if env_file.exists() else None


def machine_env(path: Optional[Path] = None) -> Dict[str, str]:
    """The DOTENV_KEYS and PYTHON_* values of .env (`KEY=value` or `export
    KEY="value"` lines)."""
    path = path or dotenv_path()
    out: Dict[str, str] = {}
    if path is None:
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (x.strip() for x in line.split("=", 1))
        if key in DOTENV_KEYS or key.startswith("PYTHON_"):
            out[key] = value.strip("\"'")
    return out


# -----------------------------------------------------------------------------
# jobs


@dataclass
class Job:
    """One command of an experiment and its scheduling state.

    Attributes:
        name: Unique name; other jobs list it in `needs`.
        cmd: Command line, run from the repository root.
        env: Environment variables layered over the scheduler's environment.
        needs: Names of jobs that must have finished first.
        gpu: Whether the job needs a GPU (else it runs on the CPU).
        evaluation: Explainer run, counted against --max-eval-jobs.
        copy_from: Run whose step-1/2 files are copied in before the start.
        copy_to: Destination run of copy_from.
        copy_crops: Also copy the step-2 crops (crops.json).
        reuse_run: Finished run whose steps 1-5 this run (from step 6) uses.
        wait_for: File that must exist before the job starts (written by
            another runner).
        state: pending, running, done or failed.
        attempts: Starts so far.
        not_before: Earliest start time (after a failed attempt).
        device: The job's device while running (cuda:N, or None on the CPU).
        proc: The running process.
    """

    name: str
    cmd: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    needs: Set[str] = field(default_factory=set)
    gpu: bool = True
    evaluation: bool = False  # explainer run: counts against --max-eval-jobs
    copy_from: Optional[Path] = None  # copy step-1/2 files from this run first
    copy_to: Optional[Path] = None
    copy_crops: bool = False
    reuse_run: Optional[Path] = (
        None  # finished run whose steps 1-5 this run (from step 6) builds on
    )
    wait_for: Optional[Path] = (
        None  # not started before this file exists (another runner writes it)
    )
    state: str = "pending"
    attempts: int = 0
    not_before: float = 0.0
    device: Optional[str] = None
    proc: Optional[subprocess.Popen] = None


def preset_env(*presets: str) -> Dict[str, str]:
    """Variables after `set -a; . configs/paper/<preset>.env ...` (later presets
    override)."""
    script = (
        "set -a; "
        + " ".join(f'. "configs/paper/{p}.env";' for p in presets)
        + " env -0"
    )
    out = subprocess.run(
        ["env", "-i", "bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    env = dict(
        item.split("=", 1) for item in out.decode().split("\0") if "=" in item
    )
    for key in ("PWD", "SHLVL", "_", "OLDPWD"):
        env.pop(key, None)
    return env


def pipeline_job(
    name: str, run: Path, presets: List[str], args: List[str] = (), **kw
) -> Job:
    """A scripts/run_full_pipeline.py job.

    Args:
        name: Job name.
        run: Output directory of the run.
        presets: configs/paper presets, in layering order (later ones win).
        args: Extra pipeline arguments (e.g. --skip-to-step 6).
        **kw: Further Job fields (needs, evaluation, copy_from, ...).

    Returns:
        The job; its environment is the merged presets.
    """
    return Job(
        name,
        [
            PY,
            "-u",
            "scripts/run_full_pipeline.py",
            "--output-dir",
            str(run),
            *args,
        ],
        env=preset_env(*presets),
        **kw,
    )


def with_variants(
    jobs: List[Job], base: str, run: Path, presets: List[str]
) -> List[str]:
    """Add the 4 evaluation variants of a base run (they reuse its concepts);
    returns all names."""
    names = [base]
    for ev in EVALS:
        jobs.append(
            pipeline_job(
                f"{base}_{ev}",
                run,
                presets + [ev],
                ["--skip-to-step", "6"],
                needs={base},
                evaluation=True,
            )
        )
        names.append(f"{base}_{ev}")
    return names


def tables_job(
    name: str, runs: List[Path], methods: str, out_dir: Path, needs: Set[str]
) -> Job:
    """A CPU job writing the paper tables of `runs` (scripts/paper_report.py).

    Args:
        name: Job name.
        runs: Run directories that go into the tables.
        methods: Comma-separated decomposition methods to report.
        out_dir: Folder of the tables.
        needs: Jobs that must have finished first.

    Returns:
        The job.
    """
    return Job(
        name,
        [
            PY,
            "scripts/paper_report.py",
            "tables",
            "--runs",
            *map(str, runs),
            "--methods",
            methods,
            "--out-dir",
            str(out_dir),
        ],
        needs=needs,
        gpu=False,
    )


COCO_METHODS = ("tgcl_sam", "tgcl_sliding_window", "coxlmm", "mcd", "craft")


def coco_jobs(out: Path, dataset: str = "coco") -> List[Job]:
    """All methods on one dataset: TGCL (SAM, sliding window), MCD, CRAFT,
    CoX-LMM.

    The baselines reuse TGCL's step-1 image lists (MCD the SAM masks, CRAFT the
    sliding windows), so every method sees the same images and segments. Each
    method also gets the 4 evaluation variants; the tables come last.

    Args:
        out: Output root; runs go to <out>/<method>_<dataset>_seed1.
        dataset: Dataset preset (coco, imagenet5, imagenet5_quick).

    Returns:
        The jobs, dependencies set.
    """
    jobs: List[Job] = []
    done: Set[str] = set()
    runs = {m: out / f"{m}_{dataset}_seed1" for m in COCO_METHODS}
    for m in ("tgcl_sam", "tgcl_sliding_window"):
        jobs.append(
            pipeline_job(m, runs[m], ["common", dataset, m], evaluation=True)
        )
        done.update(with_variants(jobs, m, runs[m], ["common", dataset, m]))
    for m, source, crops in (
        ("mcd", "tgcl_sam", True),
        ("craft", "tgcl_sliding_window", True),
        ("coxlmm", "tgcl_sliding_window", False),
    ):
        jobs.append(
            pipeline_job(
                m,
                runs[m],
                ["common", dataset, m],
                needs={source},
                evaluation=True,
                copy_from=runs[source],
                copy_to=runs[m],
                copy_crops=crops,
            )
        )
        done.update(with_variants(jobs, m, runs[m], ["common", dataset, m]))
    jobs.append(
        tables_job(
            "tables",
            list(runs.values()),
            "snmf,simple,sae2,pca",
            out / "_report" / dataset,
            done,
        )
    )
    return jobs


def coco_full_jobs(out: Path) -> List[Job]:
    """Steps 6-8 of the finished coco runs on uncropped test images
    (configs/paper/coco_full.env), in <method>_coco_full_seed1: the concepts do
    not depend on the test images, so they are reused.
    """
    jobs: List[Job] = []
    done: Set[str] = set()
    runs = []
    for m in COCO_METHODS:
        src, run = out / f"{m}_coco_seed1", out / f"{m}_coco_full_seed1"
        presets = ["common", "coco", "coco_full", m]
        jobs.append(
            pipeline_job(
                m,
                run,
                presets,
                ["--skip-to-step", "6"],
                evaluation=True,
                reuse_run=src,
            )
        )
        done.update(with_variants(jobs, m, run, presets))
        runs.append(run)
    jobs.append(
        tables_job(
            "tables",
            runs,
            "snmf,simple,sae2,pca",
            out / "_report" / "coco_full",
            done,
        )
    )
    return jobs


def coco_raw_crops_jobs(out: Path) -> List[Job]:
    """TGCL on COCO with segments passed as cut (configs/paper/raw_crops.env),
    tested on the uncropped images (coco_full.env), in
    <method>_coco_raw_crops_seed1: steps 4-8, reusing the coco runs' images and
    segments (steps 1-2)."""
    jobs: List[Job] = []
    done: Set[str] = set()
    runs = []
    for m in ("tgcl_sam", "tgcl_sliding_window"):
        src, run = out / f"{m}_coco_seed1", out / f"{m}_coco_raw_crops_seed1"
        presets = ["common", "coco", "coco_full", m, "raw_crops"]
        jobs.append(
            pipeline_job(
                m,
                run,
                presets,
                ["--skip-to-step", "4"],
                evaluation=True,
                copy_from=src,
                copy_to=run,
                copy_crops=True,
            )
        )
        done.update(with_variants(jobs, m, run, presets))
        runs.append(run)
    jobs.append(
        tables_job(
            "tables", runs, "snmf", out / "_report" / "coco_raw_crops", done
        )
    )
    return jobs


# Dataset presets of the TGCL experiment (layered after common.env). COCO is
# tested on the uncropped
# images (coco_full.env); run directories are <config>_<dataset>_seed1.
TGCL_DATASETS = {
    "coco": ["coco", "coco_full"],
    "coco_detected": ["coco", "coco_detected"],
    "imagenet5": ["imagenet5"],
    "imagenet5_quick": ["imagenet5_quick"],
    "imagenet1k_quick": ["imagenet1k_quick"],
}
# Datasets that only change the test images: their runs take steps 1-5
# (concepts) from these.
TGCL_REUSE = {"coco_detected": "coco"}
TGCL_CONFIGS = ("tgcl_sam", "tgcl_sliding_window")
# Configs whose step-2 segments are worth taking from another model's run
# (--reuse-segments-from):
# SAM masks are class-free and model-free, and cost ~1 s per image; sliding
# windows cost nothing.
TGCL_SEGMENT_REUSE = ("tgcl_sam",)


def tgcl_jobs(
    out: Path,
    dataset: str,
    variants: bool = True,
    segments_from: Optional[Path] = None,
) -> List[Job]:
    """TGCL only, both segment types: all steps, the 4 evaluation variants
    (variants=False: none) and the tables, with the snmf decomposition of the
    paper. segments_from: output
    root of another model's runs of this dataset (e.g. outputs/paper); its SAM
    run's step-1/2 files are copied once its crops.json exists, so SAM runs once
    for all models.
    """
    jobs: List[Job] = []
    done: Set[str] = set()
    runs = []
    for cfg in TGCL_CONFIGS:
        run = out / f"{cfg}_{dataset}_seed1"
        presets = ["common", *TGCL_DATASETS[dataset], cfg]
        if (
            # Same concepts, other test images: steps 6-8 on the source's 1-5.
            dataset
            in TGCL_REUSE
        ):
            jobs.append(
                pipeline_job(
                    cfg,
                    run,
                    presets,
                    ["--skip-to-step", "6"],
                    evaluation=True,
                    reuse_run=out / f"{cfg}_{TGCL_REUSE[dataset]}_seed1",
                )
            )
        elif (
            segments_from is not None
            and cfg in TGCL_SEGMENT_REUSE
            and (segments_from / run.name).resolve() != run.resolve()
        ):
            src = segments_from / run.name
            jobs.append(
                pipeline_job(
                    cfg,
                    run,
                    presets,
                    evaluation=True,
                    copy_from=src,
                    copy_to=run,
                    copy_crops=True,
                    wait_for=src / "inference" / "crops.json",
                )
            )
        else:
            jobs.append(pipeline_job(cfg, run, presets, evaluation=True))
        names = with_variants(jobs, cfg, run, presets) if variants else [cfg]
        for job in jobs:
            if job.name in names:
                job.env["DECOMP_METHODS"] = "snmf"
        done.update(names)
        runs.append(run)
    jobs.append(
        tables_job(
            "tables", runs, "snmf", out / "_report" / f"tgcl_{dataset}", done
        )
    )
    return jobs


def imagenet1k_jobs(
    out: Path,
    shards: int,
    extra: Optional[Dict[str, str]] = None,
    reuse_segments: Optional[Path] = None,
) -> List[Job]:
    """reuse_segments: an earlier imagenet1k output root with the same --shards.
    Its step-2 segments (crops.json) are copied, so step 2 is skipped: they do
    not depend on the model (class-free SAM masks, sliding windows; ImageNet
    classes come from the labels).
    """
    jobs: List[Job] = []
    done: Set[str] = set()
    runs = []
    for cfg in ("tgcl_sliding_window", "tgcl_sam"):
        run = out / f"{cfg}_imagenet1k_seed1"
        runs.append(run)
        presets = ["common", "imagenet1k", cfg]
        shard_names = []
        for k in range(shards):
            name = f"{cfg}_shard{k}of{shards}"
            shard_run = out / f"{cfg}_imagenet1k_seed1_shard{k}of{shards}"
            env = {
                "DECOMP_METHODS": "snmf",
                "CONCEPT_SHARD": f"{k}/{shards}",
                **(extra or {}),
            }
            needs = set()
            reuse = (
                dict(
                    copy_from=reuse_segments / shard_run.name,
                    copy_to=shard_run,
                    copy_crops=True,
                )
                if reuse_segments
                else {}
            )
            if (
                cfg == "tgcl_sliding_window" and not reuse
            ):  # sliding-window crops need no GPU
                crops = pipeline_job(
                    f"{name}_crops",
                    shard_run,
                    presets,
                    ["--stop-after-step", "2"],
                    gpu=False,
                )
                crops.env.update({**env, "DEVICE": "cpu"})
                jobs.append(crops)
                needs = {crops.name}
            job = pipeline_job(
                name,
                shard_run,
                presets,
                ["--stop-after-step", "5"],
                needs=needs,
                **reuse,
            )
            job.env.update(env)
            jobs.append(job)
            shard_names.append(name)
        merge = Job(
            f"{cfg}_merge",
            [
                PY,
                "scripts/merge_concept_shards.py",
                "--methods",
                "snmf",
                "--out",
                str(run),
                "--shards",
                *[
                    str(out / f"{cfg}_imagenet1k_seed1_shard{k}of{shards}")
                    for k in range(shards)
                ],
            ],
            needs=set(shard_names),
            gpu=False,
        )
        jobs.append(merge)
        jobs.append(
            pipeline_job(
                cfg,
                run,
                presets,
                ["--skip-to-step", "6"],
                needs={merge.name},
                evaluation=True,
            )
        )
        names = with_variants(jobs, cfg, run, presets)
        for job in jobs:
            if job.name in names:  # the merged run's evaluations: snmf only
                job.env.update({"DECOMP_METHODS": "snmf", **(extra or {})})
        done.update(names)
    jobs.append(
        tables_job("tables", runs, "snmf", out / "_report" / "imagenet1k", done)
    )
    return jobs


# -----------------------------------------------------------------------------
# GPUs


def gpu_memory() -> Dict[int, int]:
    """{gpu index: MiB used}."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        int(i): int(m)
        for i, m in (line.split(",") for line in out.strip().splitlines())
    }


def pipeline_gpus() -> Set[int]:
    """GPUs in the DEVICE of any running pipeline process (also one between
    two steps)."""
    used = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = (
                open(f"/proc/{pid}/cmdline", "rb")
                .read()
                .decode(errors="replace")
            )
            if "run_full_pipeline.py" not in cmd:
                continue
            env = (
                open(f"/proc/{pid}/environ", "rb")
                .read()
                .decode(errors="replace")
                .split("\0")
            )
        except OSError:
            continue
        for item in env:
            if item.startswith("DEVICE=cuda:"):
                used.add(int(item.split(":")[1]))
    return used


# Runners (one per model / dataset) pick GPUs one at a time: a started job's
# process already carries its
# DEVICE, so the next runner sees that GPU in pipeline_gpus(). Without the lock,
# runners polling at the same
# moment took the same just-freed GPU and their models ran out of memory
# together.
GPU_CLAIM_LOCK = Path(
    os.environ.get(
        "RUN_EXPERIMENTS_GPU_LOCK",
        Path(tempfile.gettempdir()) / "run_experiments_gpu_claim.lock",
    )
)


@contextlib.contextmanager
def gpu_claim_lock(path: Path = GPU_CLAIM_LOCK):
    """Exclusive across processes (fcntl); released by the OS if the holder
    dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def free_gpus(
    allowed: Optional[Set[int]], free_mib: int, busy: Set[int]
) -> List[int]:
    """GPUs a new job may take.

    Args:
        allowed: GPU indices to choose from (None: all).
        free_mib: A GPU counts as free below this memory use (MiB).
        busy: GPUs already given to this scheduler's jobs.

    Returns:
        Free GPU indices, ascending; GPUs assigned to any pipeline process (its
        DEVICE) are excluded even while they still show no memory use.
    """
    memory = gpu_memory()
    taken = busy | pipeline_gpus()
    return [
        g
        for g, used in sorted(memory.items())
        if used < free_mib
        and g not in taken
        and (allowed is None or g in allowed)
    ]


# -----------------------------------------------------------------------------
# scheduler


def copy_inputs(job: Job) -> None:
    """Step-1 (and with copy_crops step-2) outputs of the source run, so the run
    reuses them."""
    src, dst = job.copy_from / "inference", job.copy_to / "inference"
    dst.mkdir(parents=True, exist_ok=True)
    names = list(STEP1_FILES) + [
        p.name for p in src.glob("concepts_to_images_*.json")
    ]
    if job.copy_crops:
        names.append("crops.json")
    for name in names:
        if (src / name).exists() and not (dst / name).exists():
            shutil.copyfile(src / name, dst / name)


# Steps 6-8 only read these of a finished run: linked (large) or copied (small,
# so they can be written).
REUSE_LINKED = ("features", "inference")
REUSE_COPIED = ("concept", "intermediate")


def reuse_finished_run(src: Path, dst: Path) -> None:
    """The step 1-5 outputs of src in dst, for a run that starts at step 6; src
    stays untouched."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in REUSE_LINKED:
        if (src / name).exists() and not (dst / name).exists():
            (dst / name).symlink_to((src / name).resolve())
    for name in REUSE_COPIED:
        if (src / name).exists() and not (dst / name).exists():
            shutil.copytree(src / name, dst / name, symlinks=True)


def thread_env(threads: int) -> Dict[str, str]:
    """CPU thread pools of one job. Each library otherwise sizes its pool to all
    cores: six jobs on a 64-core machine ran ~100 threads each (load 100) and
    starved their GPUs.
    """
    n = str(max(1, threads))
    return {
        "OMP_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "TOKENIZERS_PARALLELISM": "false",
    }


class Scheduler:
    """Runs jobs in dependency order: one job per GPU, retries, capped
    evaluations.

    A pending job starts once its `needs` are done, its retry delay has passed
    and its `wait_for` file exists. GPU jobs take free GPUs under a
    cross-process lock; at most `max_eval_jobs` explainer evaluations run at
    once (each holds its results in host RAM). A failed job is retried up to
    `max_attempts` times; jobs that need it are then skipped.
    """

    def __init__(
        self,
        jobs: List[Job],
        logs: Path,
        gpus: Optional[Set[int]],
        max_gpus: int,
        free_mib: int,
        max_cpu_jobs: int,
        max_eval_jobs: int,
        max_attempts: int,
        retry_s: int,
        poll_s: int,
        threads_per_job: int = 0,
    ):
        self.jobs, self.logs = jobs, logs
        self.threads = (
            thread_env(threads_per_job) if threads_per_job > 0 else {}
        )
        self.machine = machine_env()
        self.gpus, self.max_gpus, self.free_mib = gpus, max_gpus, free_mib
        self.max_cpu, self.max_eval = max_cpu_jobs, max_eval_jobs
        self.max_attempts, self.retry_s, self.poll_s = (
            max_attempts,
            retry_s,
            poll_s,
        )
        self.by_name = {j.name: j for j in jobs}
        self._announced_waits: Set[str] = set()
        unknown = {n for j in jobs for n in j.needs} - set(self.by_name)
        if unknown:
            raise ValueError(f"unknown dependencies: {sorted(unknown)}")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(
            self.logs / "run_experiments.log", "a", encoding="utf-8"
        ) as f:
            f.write(line + "\n")

    def start(self, job: Job, device: Optional[str]) -> None:
        """Start a job on a device (None: the CPU, with no GPU visible).

        Copies or links the inputs it reuses first; its output goes to
        <logs>/<job name>.log.
        """
        if job.copy_from is not None:
            copy_inputs(job)
        if job.reuse_run is not None:
            reuse_finished_run(
                job.reuse_run, Path(job.cmd[job.cmd.index("--output-dir") + 1])
            )
        # explicit settings win: .env < this process's environment < the presets
        env = {
            **self.threads,
            **self.machine,
            **os.environ,
            **job.env,
            "PYTHONUNBUFFERED": "1",
            "XLVLMS_DOTENV": "0",
        }
        if device:
            env["DEVICE"] = device
        else:  # CPU job: never touch a GPU
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["DEVICE"] = "cpu"
        job.device, job.attempts, job.state = (
            device,
            job.attempts + 1,
            "running",
        )
        self.logs.mkdir(parents=True, exist_ok=True)
        job.proc = subprocess.Popen(
            job.cmd,
            cwd=ROOT,
            env=env,
            stdout=open(self.logs / f"{job.name}.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        self.log(f"START {job.name} on {device or 'cpu'}")

    def poll(self) -> None:
        """Record finished jobs: done, pending again (retry) or failed.

        A pending job that needs a failed job is marked failed (skipped).
        """
        for job in self.jobs:
            if job.state == "running" and job.proc.poll() is not None:
                if job.proc.returncode == 0:
                    job.state = "done"
                    self.log(f"DONE  {job.name}")
                elif job.attempts < self.max_attempts:
                    job.state, job.not_before = (
                        "pending",
                        time.time() + self.retry_s,
                    )
                    self.log(
                        f"RETRY {job.name} (attempt {job.attempts} "
                        f"failed; see _logs/{job.name}.log)"
                    )
                else:
                    job.state = "failed"
                    self.log(f"FAIL  {job.name} after {job.attempts} attempts")
        failed = {j.name for j in self.jobs if j.state == "failed"}
        for job in self.jobs:
            if job.state == "pending" and job.needs & failed:
                job.state = "failed"
                self.log(
                    f"SKIP  {job.name} (needs {sorted(job.needs & failed)})"
                )

    def ready(self) -> List[Job]:
        """Pending jobs whose dependencies, retry delay and wait_for file allow
        a start (a job waiting for its file is logged once)."""
        now = time.time()
        jobs = [
            j
            for j in self.jobs
            if j.state == "pending"
            and j.not_before <= now
            and all(self.by_name[n].state == "done" for n in j.needs)
        ]
        waiting = [
            j
            for j in jobs
            if j.wait_for is not None and not j.wait_for.exists()
        ]
        for job in waiting:
            if job.name not in self._announced_waits:
                self._announced_waits.add(job.name)
                self.log(f"WAIT  {job.name} (for {job.wait_for})")
        return [j for j in jobs if j not in waiting]

    def step(self) -> None:
        """One scheduling round: poll, then start ready CPU jobs and GPU jobs.

        CPU jobs fill free CPU slots; GPU jobs take free GPUs (claimed under the
        cross-process lock) within --max-gpus and the evaluation cap.
        """
        self.poll()
        ready = self.ready()
        running = [j for j in self.jobs if j.state == "running"]
        cpu_slots = self.max_cpu - sum(1 for j in running if not j.gpu)
        for job in [j for j in ready if not j.gpu][: max(0, cpu_slots)]:
            self.start(job, None)
        busy = {int(j.device.split(":")[1]) for j in running if j.device}
        evals = sum(1 for j in running if j.evaluation)
        gpu_jobs = []
        for job in (j for j in ready if j.gpu):
            if job.evaluation:
                if evals >= self.max_eval:
                    continue
                evals += 1
            gpu_jobs.append(job)
        if gpu_jobs:
            with gpu_claim_lock():
                gpus = free_gpus(self.gpus, self.free_mib, busy)[
                    : max(0, self.max_gpus - len(busy))
                ]
                for job, g in zip(gpu_jobs, gpus):
                    self.start(job, f"cuda:{g}")

    def run(self) -> bool:
        self.log(f"{len(self.jobs)} jobs")
        while any(j.state in ("pending", "running") for j in self.jobs):
            self.step()
            time.sleep(self.poll_s)
        failed = [j.name for j in self.jobs if j.state == "failed"]
        self.log("ALL DONE" + (f" with failures: {failed}" if failed else ""))
        return not failed


def select(jobs: List[Job], only: str) -> List[Job]:
    """Jobs whose name starts with one of the comma-separated prefixes; their
    other dependencies are taken as done (their run directories must exist)."""
    prefixes = tuple(p for p in only.split(",") if p)
    kept = [j for j in jobs if j.name.startswith(prefixes)]
    names = {j.name for j in kept}
    for job in kept:
        job.needs &= names
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "experiment",
        choices=[
            "coco",
            "coco_full",
            "coco_raw_crops",
            "imagenet5",
            "imagenet5_quick",
            "tgcl",
            "imagenet1k",
        ],
    )
    ap.add_argument(
        "--dataset",
        choices=sorted(TGCL_DATASETS),
        default="imagenet5",
        help="tgcl: dataset",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default outputs/paper, with --model outputs/paper_<model>",
    )
    ap.add_argument(
        "--model",
        default="",
        help="model preset configs/paper/<model>.env layered on every run "
        "(default: Gemma-3n "
        "of common.env), e.g. qwen2_5_vl_7b",
    )
    ap.add_argument(
        "--shards",
        type=int,
        default=8,
        help="imagenet1k: class shards per crop mode",
    )
    ap.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU indices to use (default: all)",
    )
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument(
        "--free-mib",
        type=int,
        default=2000,
        help="a GPU counts as free below this memory use",
    )
    ap.add_argument("--max-cpu-jobs", type=int, default=8)
    ap.add_argument(
        "--max-eval-jobs",
        type=int,
        default=2,
        help="concurrent explainer evaluations (host RAM)",
    )
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument(
        "--threads-per-job",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 8),
        help="CPU threads per job (OMP/MKL/OpenBLAS; 0 = library defaults)",
    )
    ap.add_argument("--retry-s", type=int, default=600)
    ap.add_argument(
        "--poll-s",
        type=int,
        default=5,
        help="seconds between checks for finished jobs and free GPUs (a check "
        "is cheap)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the jobs and their dependencies, run nothing",
    )
    ap.add_argument(
        "--reuse-segments-from",
        type=Path,
        default=None,
        help="imagenet1k: earlier output root (same --shards) whose step-2 "
        "segments are reused; "
        "tgcl: output root of another model (e.g. outputs/paper) whose "
        "SAM segments the "
        "SAM run copies, waiting until they exist",
    )
    ap.add_argument(
        "--no-variants",
        action="store_true",
        help="tgcl: base evaluation only (single test images), no "
        "grid/text/image/combined variants",
    )
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated job-name prefixes to run (e.g. 'craft,tables'); "
        "other jobs count as done",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny sizes (a few images per class) into <out>_smoke: checks "
        "every stage in minutes",
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = (
            ROOT
            / "outputs"
            / (f"paper_{args.model}" if args.model else "paper")
        )
    if args.smoke:
        args.out = args.out.with_name(args.out.name + "_smoke")
    if args.experiment == "tgcl":
        jobs = tgcl_jobs(
            args.out,
            args.dataset,
            variants=not args.no_variants,
            segments_from=args.reuse_segments_from.resolve()
            if args.reuse_segments_from
            else None,
        )
        missing = [
            str(j.reuse_run)
            for j in jobs
            if j.reuse_run and not (j.reuse_run / "concept").is_dir()
        ]
        if missing:
            sys.exit(
                f"tgcl --dataset {args.dataset} reuses finished runs' "
                "concepts; run "
                f"--dataset {TGCL_REUSE[args.dataset]} first: {missing}"
            )
    elif args.experiment in ("coco", "imagenet5", "imagenet5_quick"):
        jobs = coco_jobs(args.out, args.experiment)
    elif args.experiment == "coco_full":
        jobs = coco_full_jobs(args.out)
        missing = [
            str(j.reuse_run)
            for j in jobs
            if j.reuse_run and not (j.reuse_run / "concept").is_dir()
        ]
        if missing:
            sys.exit(
                "coco_full needs the finished coco runs (python "
                f"scripts/run_experiments.py coco): {missing}"
            )
    elif args.experiment == "coco_raw_crops":
        jobs = coco_raw_crops_jobs(args.out)
        missing = [
            str(j.copy_from)
            for j in jobs
            if j.copy_from
            and not (j.copy_from / "inference/crops.json").exists()
        ]
        if missing:
            sys.exit(
                "coco_raw_crops needs the coco runs' segments (python "
                f"scripts/run_experiments.py coco): {missing}"
            )
    else:
        jobs = imagenet1k_jobs(
            args.out, args.shards, reuse_segments=args.reuse_segments_from
        )
    if args.model:
        model_env = preset_env(args.model)
        for job in jobs:
            if "scripts/run_full_pipeline.py" in job.cmd:
                job.env.update(model_env)
    if args.only:
        jobs = select(jobs, args.only)
    if args.smoke:
        for job in jobs:
            job.env.update(SMOKE_ENV)
            if "DECOMP_METHODS" in job.env:
                job.env["DECOMP_METHODS"] = ",".join(
                    m
                    for m in job.env["DECOMP_METHODS"].split(",")
                    if m not in SMOKE_SKIP_METHODS
                )
            if "--methods" in job.cmd:
                i = job.cmd.index("--methods") + 1
                job.cmd[i] = ",".join(
                    m
                    for m in job.cmd[i].split(",")
                    if m not in SMOKE_SKIP_METHODS
                )
    if args.dry_run:
        for job in jobs:
            needs = f"  <- {', '.join(sorted(job.needs))}" if job.needs else ""
            print(f"{'GPU' if job.gpu else 'CPU'}  {job.name}{needs}")
        print(f"{len(jobs)} jobs")
        return
    gpus = {int(g) for g in args.gpus.split(",")} if args.gpus else None
    log_name = (
        f"tgcl_{args.dataset}" if args.experiment == "tgcl" else args.experiment
    )
    ok = Scheduler(
        jobs,
        args.out / "_logs" / log_name,
        gpus,
        args.max_gpus,
        args.free_mib,
        args.max_cpu_jobs,
        args.max_eval_jobs,
        args.max_attempts,
        args.retry_s,
        args.poll_s,
        args.threads_per_job,
    ).run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
