"""scripts/run_experiments.py: job graphs, GPU choice, evaluation cap, retries,
input reuse."""

import contextlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rx():
    spec = importlib.util.spec_from_file_location(
        "run_experiments", REPO / "tgcl" / "experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiments"] = module
    spec.loader.exec_module(module)
    return module


def test_coco_graph(rx, tmp_path):
    jobs = {j.name: j for j in rx.coco_jobs(tmp_path)}
    assert len(jobs) == 26  # 5 methods x (base + 4 variants) + tables
    assert (
        jobs["mcd"].needs == {"tgcl_sam"} and jobs["mcd"].copy_crops
    )  # SAM masks reused
    assert (
        jobs["craft"].needs == {"tgcl_sliding_window"}
        and jobs["craft"].copy_crops
    )
    assert (
        jobs["coxlmm"].copy_from.name == "tgcl_sliding_window_coco_seed1"
        and not jobs["coxlmm"].copy_crops
    )
    assert (
        jobs["mcd_eval_grid"].needs == {"mcd"}
        and "--skip-to-step" in jobs["mcd_eval_grid"].cmd
    )
    assert jobs["mcd"].env["DECOMP_METHODS"] == "pca"  # MCD preset
    assert (
        jobs["tgcl_sam"].env["CONCEPT_SPARSIFY"] == "magnitude:0.5"
    )  # TGCL preset
    assert jobs["craft"].env["BATCH_SIZE"] == "16"  # fits a 24 GB GPU
    assert (
        jobs["tables"].needs == set(jobs) - {"tables"}
        and not jobs["tables"].gpu
    )


def test_imagenet1k_graph(rx, tmp_path):
    jobs = {j.name: j for j in rx.imagenet1k_jobs(tmp_path, shards=2)}
    assert not jobs[
        "tgcl_sliding_window_shard1of2_crops"
    ].gpu  # crops on the CPU
    assert jobs["tgcl_sliding_window_shard1of2"].needs == {
        "tgcl_sliding_window_shard1of2_crops"
    }
    assert jobs["tgcl_sam_shard0of2"].env["CONCEPT_SHARD"] == "0/2"
    assert jobs["tgcl_sam_merge"].needs == {
        "tgcl_sam_shard0of2",
        "tgcl_sam_shard1of2",
    }
    assert (
        jobs["tgcl_sam"].needs == {"tgcl_sam_merge"}
        and jobs["tgcl_sam_eval_combined"].env["DECOMP_METHODS"] == "snmf"
    )


def test_free_gpus_skip_busy_and_pipeline_assigned(rx, monkeypatch):
    monkeypatch.setattr(
        rx, "gpu_memory", lambda: {0: 10, 1: 10, 2: 5000, 3: 10, 4: 10}
    )
    monkeypatch.setattr(
        rx, "pipeline_gpus", lambda: {3}
    )  # a run between two steps keeps its GPU
    assert rx.free_gpus(None, 2000, busy={1}) == [0, 4]
    assert rx.free_gpus({0, 2}, 2000, busy=set()) == [0]


def _scheduler(rx, jobs, tmp_path, **kw):
    opts = dict(
        gpus=None,
        max_gpus=4,
        free_mib=2000,
        max_cpu_jobs=4,
        max_eval_jobs=1,
        max_attempts=2,
        retry_s=0,
        poll_s=0,
    )
    opts.update(kw)
    return rx.Scheduler(jobs, tmp_path / "_logs", **opts)


def test_eval_cap_and_gpu_assignment(rx, tmp_path, monkeypatch):
    monkeypatch.setattr(
        rx,
        "free_gpus",
        lambda allowed, mib, busy: [g for g in (0, 1, 2) if g not in busy],
    )
    started = []
    jobs = [
        rx.Job("a", [], evaluation=True),
        rx.Job("b", [], evaluation=True),
        rx.Job("c", []),
    ]
    s = _scheduler(rx, jobs, tmp_path)
    monkeypatch.setattr(
        s,
        "start",
        lambda job, device: (
            started.append((job.name, device)),
            setattr(job, "state", "running"),
            setattr(job, "device", device),
        ),
    )
    s.step()
    assert started == [
        ("a", "cuda:0"),
        ("c", "cuda:1"),
    ]  # b waits: one evaluation at a time


def test_cpu_jobs_run_retry_and_skip_dependents(rx, tmp_path):
    flag = tmp_path / "second_try"
    flaky = (
        f"import pathlib, sys; p = pathlib.Path({str(flag)!r}); ok = "
        "p.exists(); p.touch(); sys.exit(0 if ok else 1)"
    )
    jobs = [
        rx.Job("flaky", [sys.executable, "-c", flaky], gpu=False),
        rx.Job(
            "broken", [sys.executable, "-c", "raise SystemExit(3)"], gpu=False
        ),
        rx.Job(
            "after", [sys.executable, "-c", "pass"], needs={"broken"}, gpu=False
        ),
    ]
    assert _scheduler(rx, jobs, tmp_path).run() is False
    state = {j.name: (j.state, j.attempts) for j in jobs}
    assert state == {
        "flaky": ("done", 2),
        "broken": ("failed", 2),
        "after": ("failed", 0),
    }
    assert (
        "SKIP  after"
        in (tmp_path / "_logs" / "run_experiments.log").read_text()
    )


def test_copy_inputs_reuses_step1_and_crops(rx, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "inference").mkdir(parents=True)
    for name in (
        "objects.csv",
        "concepts_to_images.json",
        "concepts_to_images_vocab_top10.json",
        "crops.json",
    ):
        (src / "inference" / name).write_text(name)
    rx.copy_inputs(
        rx.Job("x", [], copy_from=src, copy_to=dst, copy_crops=False)
    )
    assert sorted(p.name for p in (dst / "inference").iterdir()) == [
        "concepts_to_images.json",
        "concepts_to_images_vocab_top10.json",
        "objects.csv",
    ]
    rx.copy_inputs(rx.Job("x", [], copy_from=src, copy_to=dst, copy_crops=True))
    assert (dst / "inference" / "crops.json").exists()


def test_unknown_dependency_is_an_error(rx, tmp_path):
    with pytest.raises(ValueError, match="unknown dependencies"):
        _scheduler(rx, [rx.Job("a", [], needs={"nope"})], tmp_path)


def test_only_runs_a_subset_and_drops_outside_dependencies(rx, tmp_path):
    jobs = {
        j.name: j for j in rx.select(rx.coco_jobs(tmp_path), "craft,tables")
    }
    assert set(jobs) == {
        "craft",
        "craft_eval_grid",
        "craft_eval_text_only",
        "craft_eval_image_only",
        "craft_eval_combined",
        "tables",
    }
    assert jobs["craft"].needs == set()  # tgcl_sliding_window: taken as done
    assert jobs["craft_eval_grid"].needs == {"craft"}
    assert jobs["tables"].needs == set(jobs) - {"tables"}


def test_imagenet1k_reuses_segments(rx, tmp_path):
    jobs = {
        j.name: j
        for j in rx.imagenet1k_jobs(
            tmp_path / "new", shards=2, reuse_segments=tmp_path / "old"
        )
    }
    assert "tgcl_sliding_window_shard0of2_crops" not in jobs  # no crop jobs
    shard = jobs["tgcl_sam_shard1of2"]
    assert (
        shard.copy_crops
        and shard.copy_from
        == tmp_path / "old" / "tgcl_sam_imagenet1k_seed1_shard1of2"
    )
    assert (
        shard.copy_to
        == tmp_path / "new" / "tgcl_sam_imagenet1k_seed1_shard1of2"
    )


def test_coco_full_graph_reuses_coco_concepts(rx, tmp_path):
    jobs = {j.name: j for j in rx.coco_full_jobs(tmp_path)}
    assert len(jobs) == 26
    base = jobs["tgcl_sam"]
    assert (
        base.reuse_run == tmp_path / "tgcl_sam_coco_seed1"
        and "--skip-to-step" in base.cmd
    )
    assert str(tmp_path / "tgcl_sam_coco_full_seed1") in base.cmd
    assert (
        base.env["IMAGE_ROOT"] == "data/coco_common10/full_test/test.json"
    )  # uncropped test images
    assert (
        jobs["tgcl_sam_eval_grid"].env["IMAGE_ROOT"]
        == "data/coco_common10/full_test/grids.json"
    )
    assert (
        jobs["tgcl_sam_eval_combined"].env["IMAGE_ROOT"]
        == "data/coco_common10/full_test/test.json"
    )
    assert jobs["mcd"].env["DECOMP_METHODS"] == "pca" and not jobs["mcd"].needs


def test_coco_raw_crops_graph(rx, tmp_path):
    jobs = {j.name: j for j in rx.coco_raw_crops_jobs(tmp_path)}
    assert len(jobs) == 11  # 2 TGCL configs x (base + 4 variants) + tables
    base = jobs["tgcl_sam"]
    assert (
        base.copy_from == tmp_path / "tgcl_sam_coco_seed1" and base.copy_crops
    )  # same segments
    assert base.cmd[-2:] == ["--skip-to-step", "4"]
    assert (
        base.env["CROP_RESIZE"] == "none"
        and base.env["QWEN_MIN_PIXELS"] == "200704"
    )
    assert (
        base.env["IMAGE_ROOT"] == "data/coco_common10/full_test/test.json"
    )  # uncropped test images
    assert base.env["PATCH_SIZE"] == "160"  # sliding windows unchanged


def test_imagenet5_graph_and_model_presets(rx, tmp_path):
    jobs = {j.name: j for j in rx.coco_jobs(tmp_path, "imagenet5")}
    assert len(jobs) == 26 and jobs["tgcl_sam"].cmd[-2:] != [
        "--skip-to-step",
        "6",
    ]
    env = jobs["tgcl_sam"].env
    assert env["CONCEPT_MAP"].endswith(
        "subsets/imagenet5/concepts_to_images.json"
    )
    assert env["IMAGE_ROOT"].endswith(
        "subsets/imagenet5/test.json"
    )  # JSON image list
    assert (
        jobs["tgcl_sam_eval_grid"]
        .env["IMAGE_ROOT"]
        .endswith("subsets/imagenet5/grids.json")
    )
    assert (
        env["PROMPT_TEMPLATE"] == "cgdl_contains"
        and "FORCED_RESPONSE" not in env
    )  # contrastive answer feature
    assert (
        env["EXPL_CENTER"] == "calibration"
        and env["SEGMENT_SELECTION"] == "largest"
    )
    assert (
        rx.preset_env("tgcl_sam", "qwen2_5_vl_7b")["PROMPT_TEMPLATE"]
        == "cgdl_anywhere"
    )
    assert (
        rx.preset_env("tgcl_sam", "qwen2_vl_7b")["VLM_MODEL"]
        == "Qwen/Qwen2-VL-7B-Instruct"
    )


def test_coco_detected_reuses_coco_concepts(rx, tmp_path):
    jobs = {j.name: j for j in rx.tgcl_jobs(tmp_path, "coco_detected")}
    assert len(jobs) == 11
    base = jobs["tgcl_sam"]
    assert (
        base.cmd[-2:] == ["--skip-to-step", "6"]
        and base.reuse_run == tmp_path / "tgcl_sam_coco_seed1"
    )
    assert (
        base.env["IMAGE_ROOT"] == "data/coco_common10/detected_test/test.json"
    )  # detected crops
    assert (
        jobs["tgcl_sam_eval_grid"].env["IMAGE_ROOT"]
        == "data/coco_common10/detected_test/grids.json"
    )
    assert (
        jobs["tables"]
        .cmd[jobs["tables"].cmd.index("--out-dir") + 1]
        .endswith("tgcl_coco_detected")
    )


def test_reuse_finished_run_links_and_copies(rx, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for d in (
        "features",
        "inference",
        "concept/snmf",
        "intermediate",
        "explanations",
    ):
        (src / d).mkdir(parents=True)
    (src / "concept/snmf/bank.pth").write_text("bank")
    rx.reuse_finished_run(src, dst)
    assert (dst / "features").is_symlink() and (dst / "inference").is_symlink()
    assert (
        not (dst / "concept").is_symlink()
        and (dst / "concept/snmf/bank.pth").read_text() == "bank"
    )
    assert not (dst / "explanations").exists()  # results are not reused
    (dst / "concept/snmf/bank.pth").write_text("changed")
    assert (
        src / "concept/snmf/bank.pth"
    ).read_text() == "bank"  # source untouched
    rx.reuse_finished_run(src, dst)  # idempotent


def test_model_preset(rx):
    env = rx.preset_env("common", "coco", "tgcl_sam", "qwen2_5_vl_7b")
    assert (
        env["VLM_MODEL"] == "Qwen/Qwen2.5-VL-7B-Instruct"
        and env["LAYER_PATH"] == "model.language_model.norm"
    )


def test_thread_cap_reaches_jobs_unless_set(rx, tmp_path, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    out = tmp_path / "env.txt"
    probe = (
        f"import os; open({str(out)!r}, "
        "'w').write(os.environ['OMP_NUM_THREADS'] + ' ' + "
        "os.environ['TOKENIZERS_PARALLELISM'])"
    )
    jobs = [rx.Job("a", [sys.executable, "-c", probe], gpu=False)]
    assert _scheduler(rx, jobs, tmp_path, threads_per_job=6).run()
    assert out.read_text() == "6 false"
    jobs = [
        rx.Job(
            "b",
            [sys.executable, "-c", probe],
            env={"OMP_NUM_THREADS": "2"},
            gpu=False,
        )
    ]
    assert _scheduler(rx, jobs, tmp_path, threads_per_job=6).run()
    assert out.read_text() == "2 false"


def test_gpus_are_claimed_under_the_cross_process_lock(
    rx, tmp_path, monkeypatch
):
    held = {"now": False}

    @contextlib.contextmanager
    def fake_lock(path=None):
        held["now"] = True
        yield
        held["now"] = False

    def fake_free(allowed, mib, busy):
        assert held["now"], "GPU picked outside the claim lock"
        return [0]

    monkeypatch.setattr(rx, "gpu_claim_lock", fake_lock)
    monkeypatch.setattr(rx, "free_gpus", fake_free)
    started = []
    sched = _scheduler(rx, [rx.Job("a", [])], tmp_path)
    monkeypatch.setattr(
        sched,
        "start",
        lambda job, device: started.append((job.name, device, held["now"])),
    )
    sched.step()
    assert started == [("a", "cuda:0", True)]


def test_claim_lock_excludes_another_process(rx, tmp_path):
    lock = tmp_path / "gpu.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, time, sys; f = open(sys.argv[1], 'a'); "
                "fcntl.flock(f, fcntl.LOCK_EX); "
                "print('held', flush=True); time.sleep(1.0)"
            ),
            str(lock),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "held"
    t0 = time.time()
    with rx.gpu_claim_lock(lock):
        waited = time.time() - t0
    holder.wait()
    assert waited > 0.5  # blocked until the other process let go


def test_imagenet1k_quick_base_only_reuses_sam_segments(rx, tmp_path):
    gemma, qwen = tmp_path / "paper", tmp_path / "paper_qwen"
    jobs = {
        j.name: j
        for j in rx.tgcl_jobs(
            qwen, "imagenet1k_quick", variants=False, segments_from=gemma
        )
    }
    assert set(jobs) == {
        "tgcl_sam",
        "tgcl_sliding_window",
        "tables",
    }  # no evaluation variants
    sam, sw = jobs["tgcl_sam"], jobs["tgcl_sliding_window"]
    assert (
        sam.copy_crops
        and sam.copy_from == gemma / "tgcl_sam_imagenet1k_quick_seed1"
    )
    assert (
        sam.wait_for
        == gemma
        / "tgcl_sam_imagenet1k_quick_seed1"
        / "inference"
        / "crops.json"
    )
    assert sw.copy_from is None and sw.wait_for is None  # sliding windows: own
    assert sam.env["NUM_CONCEPT"] == "1000" and sam.env["BAG_SIZE"] == "300"
    assert (
        sam.env["IMAGE_ROOT"]
        == "data/imagenet1k/subsets/imagenet1k_quick/test.json"
    )
    assert jobs["tables"].needs == {"tgcl_sam", "tgcl_sliding_window"}
    own = {
        j.name: j
        for j in rx.tgcl_jobs(gemma, "imagenet1k_quick", segments_from=gemma)
    }
    assert own["tgcl_sam"].wait_for is None  # its own segments


def test_machine_env_takes_only_secrets_caches_and_interpreters(rx, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nexport "
        "HF_TOKEN=\"hf_x\"\nHF_HOME='/hdd/hf'\n"
        "NUM_CONCEPT=99\nVLM_MODEL=other\n"
        "PYTHON_QWEN2_VL_7B=/venv/bin/python\nbroken line\n"
    )
    assert rx.machine_env(env) == {
        "HF_TOKEN": "hf_x",
        "HF_HOME": "/hdd/hf",
        "PYTHON_QWEN2_VL_7B": "/venv/bin/python",
    }


def test_jobs_get_dotenv_machine_values_but_presets_win(
    rx, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        rx,
        "machine_env",
        lambda path=None: {"HF_TOKEN": "hf_x", "HF_HOME": "/from/dotenv"},
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    out = tmp_path / "env.txt"
    probe = (
        f"import os; open({str(out)!r}, 'w').write(' "
        "'.join(os.environ.get(k, '-') for k in "
        f"('HF_TOKEN', 'HF_HOME', 'XLVLMS_DOTENV')))"
    )
    jobs = [
        rx.Job(
            "a",
            [sys.executable, "-c", probe],
            env={"HF_HOME": "/from/preset"},
            gpu=False,
        )
    ]
    assert _scheduler(rx, jobs, tmp_path).run()
    assert (
        out.read_text() == "hf_x /from/preset 0"
    )  # .env value passed on; preset wins; no own .env load


def test_job_waits_for_its_file(rx, tmp_path):
    marker = tmp_path / "crops.json"
    jobs = [
        rx.Job("a", [sys.executable, "-c", "pass"], gpu=False, wait_for=marker)
    ]
    s = _scheduler(rx, jobs, tmp_path)
    s.step()
    s.step()
    assert jobs[0].state == "pending"
    assert (tmp_path / "_logs" / "run_experiments.log").read_text().count(
        "WAIT  a"
    ) == 1  # logged once
    marker.write_text("{}")
    s.step()
    assert jobs[0].state == "running"
    jobs[0].proc.wait()
