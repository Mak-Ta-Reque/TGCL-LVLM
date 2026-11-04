"""scripts/evaluate_banks.py (and rerank_explanations' centring and per-class
subset) on synthetic runs with known answers."""

import json
import math

import evaluate_banks as eb
import numpy as np
import pytest
import rerank_explanations as rx
import torch

from tgcl.reporting import dictionary_metrics as dm

CLASSES = ["bench", "car", "cup"]


def _bank(path, concepts, names, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "concepts": torch.tensor(np.asarray(concepts), dtype=torch.float32),
            "concept_names": names,
            "image_grounding_paths": [[f"g{i}.jpg"] for i in range(len(names))],
            **extra,
        },
        path,
    )


def _source(tmp_path, tags=("",), per_class=2, center=None):
    """A source run: per setting, test images of 3 classes whose answer
    states point along their class's axis (plus a shared offset)."""
    run = tmp_path / "source"
    rng = np.random.default_rng(0)
    for tag in tags:
        folder = eb.explanations_dir(run, tag) / eb.SOURCE_DECOMPOSITION
        folder.mkdir(parents=True)
        acts, results = [], []
        for c, label in enumerate(CLASSES):
            for i in range(per_class):
                state = np.zeros(6)
                state[c] = 1.0
                acts.append(state + 0.05 * rng.normal(size=6) + 3.0)
                results.append(
                    {
                        "image_path": f"/data/test/{label}/{i}.jpg",
                        "per_token_concepts": [
                            {
                                "token_text": label,
                                "activation_index": len(acts) - 1,
                                "top_concepts": [],
                            }
                        ],
                    }
                )
        np.save(folder / "vlm_explanations_activations.npy", np.array(acts))
        if center is not None:
            np.save(folder / "vlm_explanations_center.npy", center)
        (folder / "vlm_explanations.json").write_text(
            json.dumps({"results": results})
        )
    return run


# ------------------------------------------------------------ rerank options


def test_first_per_class_keeps_file_order():
    results = [
        {"image_path": f"/t/{c}/{i}.jpg"} for c in "ab" for i in range(3)
    ]
    kept = rx.first_per_class(results, 2)
    assert [r["image_path"] for r in kept] == [
        "/t/a/0.jpg",
        "/t/a/1.jpg",
        "/t/b/0.jpg",
        "/t/b/1.jpg",
    ]


def test_rerank_centring_is_optional(tmp_path):
    # Concept 0 is the shared offset, concepts 1-3 the class axes: plain
    # cosine picks the offset for every image, centred matching the class.
    src = _source(tmp_path, center=np.full(6, 3.0))
    json_path = src / "explanations/snmf/vlm_explanations.json"
    concepts = np.vstack([np.full(6, 1.0), np.eye(6)[:3]])
    bank = {
        "concepts": torch.tensor(concepts),
        "concept_names": [["offset"], ["bench"], ["car"], ["cup"]],
    }
    for centred, expected in ((False, {0}), (True, {1, 2, 3})):
        out = tmp_path / f"c{centred}" / "vlm_explanations.json"
        n = rx.rerank_file(json_path, bank, out, 1, centred=centred)
        payload = json.loads(out.read_text())
        top = {
            r["per_token_concepts"][0]["top_concepts"][0]["concept_index"]
            for r in payload["results"]
        }
        assert n == 6 and top == expected
    out = tmp_path / "sub" / "vlm_explanations.json"
    assert rx.rerank_file(json_path, bank, out, 1, per_class=1) == 3


# ------------------------------------------------------------------- build


def test_merged_bank_adds_rejected_directions_and_drops_partial_fields(
    tmp_path,
):
    folder = tmp_path / "concept" / "snmf"
    _bank(
        folder / "combined_concept_snmf_cr0_raw.pth",
        np.eye(2, 6),
        [["a"], ["b"]],
        text_grounding=[["x"], ["y"]],
    )
    _bank(
        folder / "combined_negative_concept_snmf_cr0_raw.pth",
        np.eye(6)[2:5],
        [["c"], ["d"], ["e"]],
    )
    only = eb.merged_bank(tmp_path, "snmf", False)
    assert len(only["concepts"]) == 2 and only["text_grounding"] == [
        ["x"],
        ["y"],
    ]
    both = eb.merged_bank(tmp_path, "snmf", True)
    assert both["concepts"].shape == (5, 6)
    assert both["concept_names"] == [["a"], ["b"], ["c"], ["d"], ["e"]]
    assert len(both["image_grounding_paths"]) == 5
    assert "text_grounding" not in both  # missing in the negative bank
    assert eb.merged_bank(tmp_path / "missing", "snmf", True) is None


def test_processed_bank_trims_and_regrounds(monkeypatch):
    calls = []
    monkeypatch.setattr(eb, "reground", lambda bank, *a: calls.append(a))
    rng = np.random.default_rng(1)
    bank = {"concepts": torch.tensor(rng.normal(size=(4, 40)) + 1.0)}
    assert eb.processed_bank(bank, "none") is bank and not calls
    out = eb.processed_bank(bank, "lowdin+magnitude:0.3")
    assert ((out["concepts"].abs() > 0).sum(1) == 12).all()  # ceil(.3 * 40)
    assert out["post_processing"] == "lowdin+magnitude:0.3"
    assert calls == [(eb.VLM, eb.LAYER)]


def test_run_name():
    assert (
        eb.run_name("TGCL (sliding window)", "snmf", "lowdin+magnitude:0.3", 2)
        == "tgcl_sliding_window_snmf_keep0.3_seed2"
    )
    assert eb.run_name("CoX-LMM", "sae2", "none", 1) == "cox_lmm_sae2_seed1"
    assert (
        eb.run_name("TGCL (SAM)", "snmf", "lowdin", 1)
        == "tgcl_sam_snmf_lowdin_seed1"
    )
    assert (
        eb.run_name("TGCL (SAM)", "snmf", "magnitude:0.25", 1)
        == "tgcl_sam_snmf_nolowdin_keep0.25_seed1"
    )


def test_build_run_accuracy_on_all_images_subset_for_step7(tmp_path):
    src = _source(tmp_path, tags=("", "image_only"), per_class=3)
    bank = {
        "concepts": torch.tensor(np.eye(6)[:3], dtype=torch.float32),
        "concept_names": [["bench"], ["car"], ["cup"]],
    }
    meta = {"decomposition": "snmf", "centred": False, "seed": 1}
    run = tmp_path / "eval_run"
    eb.build_run(run, bank, meta, src, per_class=1)
    assert (run / "concept/snmf/combined_concept_snmf_cr0.6_raw.pth").exists()
    accuracy = json.loads((run / eb.ACCURACY).read_text())
    assert set(accuracy) == {"", "image_only"}
    assert accuracy[""]["n_images"] == 9  # every test image
    assert accuracy[""]["top1_accuracy"] == pytest.approx(1.0)
    step7 = json.loads(
        (run / "explanations_image_only/snmf/vlm_explanations.json").read_text()
    )
    assert len(step7["results"]) == 3  # one per class
    assert json.loads((run / eb.META).read_text()) == meta


# ----------------------------------------------------------- jobs and table


def _built(tmp_path, seeds=(1, 2), method="CoX-LMM", acc=(0.5, 0.7)):
    out = tmp_path / "out"
    for seed, a in zip(seeds, acc):
        run = out / eb.run_name(method, "snmf", "none", seed)
        (run / "explanations_text_only/snmf").mkdir(parents=True)
        (run / "explanations/snmf").mkdir(parents=True)
        meta = {
            "dataset": "toy",
            "method": method,
            "decomposition": "snmf",
            "post_processing": "none",
            "seed": seed,
        }
        (run / eb.META).write_text(json.dumps(meta))
        (run / eb.ACCURACY).write_text(
            json.dumps({"": {"top1_accuracy": a, "top1_f1_macro": a / 2}})
        )
    return out


def test_evaluation_jobs_skip_finished_settings(tmp_path):
    out = _built(tmp_path, seeds=(1,), acc=(0.5,))
    jobs = eb.evaluation_jobs(out, "coco")
    assert sorted(j.name for j in jobs) == [
        "cox_lmm_snmf_seed1_mcq",
        "cox_lmm_snmf_seed1_text_only",
    ]
    job = jobs[0]
    assert job.env["DECOMP_METHODS"] == "snmf" and job.env["SEED"] == "1"
    assert job.env["CONCEPT_SPARSIFY"] == ""
    assert job.cmd[-4:] == ["--skip-to-step", "7", "--stop-after-step", "7"]
    done = out / "cox_lmm_snmf_seed1" / "eval_text_only" / "snmf"
    done.mkdir(parents=True)
    (done / "clip_bert_topk_table.csv").write_text("")
    assert len(eb.evaluation_jobs(out, "coco")) == 2  # curves still missing
    (done / "concept_curves.json").write_text("{}")
    assert [j.name for j in eb.evaluation_jobs(out, "coco")] == [
        "cox_lmm_snmf_seed1_mcq"
    ]


def test_summarize_means_over_seeds_and_markdown(tmp_path):
    records = eb.summarize(_built(tmp_path))
    mcq = next(r for r in records if r["setting"] == "MCQ")
    assert mcq["seeds"] == 2
    assert mcq["acc_mean"] == pytest.approx(0.6)
    assert mcq["acc_sd"] == pytest.approx(0.1)
    assert mcq["f1_mean"] == pytest.approx(0.3)
    assert math.isnan(mcq["gap_net_mean"])  # no step-7 outputs
    table = eb.markdown(records, "MCQ")
    assert "| CoX-LMM | SNMF | 0.600 | 0.300 | – | – | – |" in table
    assert table.count("\n") == 1 + len(dm.table_rows())


def test_curve_items_pending_and_overwrite(tmp_path):
    out = _built(tmp_path, seeds=(1,), acc=(0.5,))
    run = out / "cox_lmm_snmf_seed1"
    for tag in ("", "text_only"):
        folder = eb.explanations_dir(run, tag) / "snmf"
        (folder / "vlm_explanations.json").write_text("{}")
    assert [(r.name, t, d, s) for r, t, d, s in eb.curve_items(out)] == [
        ("cox_lmm_snmf_seed1", "", "snmf", 1),
        ("cox_lmm_snmf_seed1", "text_only", "snmf", 1),
    ]
    done = eb.eval_dir(run, "") / "snmf"
    done.mkdir(parents=True)
    (done / "concept_curves.json").write_text('{"gap": 0.5}')
    assert [t for _, t, _, _ in eb.curve_items(out)] == ["text_only"]
    assert len(eb.curve_items(out, overwrite=True)) == 2
