"""Class shards, label concept maps, shard merging and sampled mcq choices."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_sparsified_shards():
    """Shards sparsified separately (CONCEPT_SPARSIFY): same setting, own
    fidelity statistics."""
    mod = _load("merge_concept_shards", "scripts/merge_concept_shards.py")
    setting = {"algorithm": "magnitude", "density": 0.25, "n_keep": 1}
    a = {
        "concepts": torch.ones(2, 4),
        "sparsification": {**setting, "score_corr": 0.99},
    }
    b = {
        "concepts": torch.zeros(3, 4),
        "sparsification": {**setting, "score_corr": 0.97},
    }
    merged = mod.merge_banks([a, b])
    assert (
        merged["concepts"].shape == (5, 4)
        and merged["sparsification"]["density"] == 0.25
    )
    assert [s["score_corr"] for s in merged["sparsification"]["per_shard"]] == [
        0.99,
        0.97,
    ]
    b["sparsification"]["density"] = 0.5
    with pytest.raises(ValueError, match="setting differs"):
        mod.merge_banks([a, b])


@pytest.fixture(scope="module")
def pipeline():
    return _load("run_full_pipeline", "scripts/run_full_pipeline.py")


@pytest.fixture(scope="module")
def merge():
    return _load("merge_concept_shards", "scripts/merge_concept_shards.py")


def test_parse_concept_shard(pipeline):
    assert pipeline.parse_concept_shard("") is None
    assert pipeline.parse_concept_shard("1/4") == (1, 4)
    with pytest.raises(ValueError):
        pipeline.parse_concept_shard("4/4")


def test_shards_partition_the_tags(pipeline, tmp_path):
    mapping = {f"class {i}": [f"img{i}.JPEG"] for i in range(10)}
    src = tmp_path / "concepts_to_images.json"
    src.write_text(json.dumps(mapping))
    parts = [
        json.loads(pipeline.shard_concept_map(src, (k, 3)).read_text())
        for k in range(3)
    ]
    assert sorted(t for p in parts for t in p) == sorted(mapping)
    assert all(p[t] == mapping[t] for p in parts for t in p)
    assert [len(p) for p in parts] == [4, 3, 3]


def _bank(names, dim=4):
    k = len(names)
    return {
        "concepts": torch.randn(k, dim),
        "concept_names": [[n] for n in names],
        "text_grounding": [["w"] for _ in names],
        "decomposition_method": "snmf",
        "clean_example_ratio": 0.6,
    }


def test_merge_banks_concatenates_in_shard_order(merge):
    a, b = _bank(["cat", "dog"]), _bank(["fish"])
    merged = merge.merge_banks([a, b])
    assert merged["concepts"].shape == (3, 4) and torch.equal(
        merged["concepts"][2], b["concepts"][0]
    )
    assert merged["concept_names"] == [["cat"], ["dog"], ["fish"]]
    assert merged["clean_example_ratio"] == 0.6
    b["clean_example_ratio"] = 0.4
    with pytest.raises(ValueError, match="differs"):
        merge.merge_banks([a, b])


def test_merge_runs_writes_banks_and_links_features(merge, tmp_path):
    shards = []
    for k, names in enumerate([["cat", "dog"], ["fish"]]):
        shard = tmp_path / f"shard{k}"
        (shard / "concept" / "snmf").mkdir(parents=True)
        torch.save(
            _bank(names),
            shard / "concept" / "snmf" / "combined_concept_snmf_cr0.6_raw.pth",
        )
        (shard / "features").mkdir()
        for n in names:
            torch.save({}, shard / "features" / f"results_{n}.pth")
        shards.append(shard)
    out = tmp_path / "merged"
    merge.merge_runs(shards, out, ["snmf"])
    bank = torch.load(
        out / "concept" / "snmf" / "combined_concept_snmf_cr0.6_raw.pth",
        weights_only=False,
    )
    assert bank["concepts"].shape[0] == 3
    assert sorted(p.name for p in (out / "features").iterdir()) == [
        "results_cat.pth",
        "results_dog.pth",
        "results_fish.pth",
    ]


def test_sampled_mcq_choices():
    from tgcl.inference.vlm_explainer_multibatch import image_classes
    from tgcl.inference.vlm_explainer_multibatch import sample_mcq_choices

    pool = [f"class {i}" for i in range(20)] + ["garbage truck"]
    assert image_classes("/val/garbage_truck/0001.JPEG") == ["garbage truck"]
    assert image_classes(
        "/g/grid_4_7__class_1__class_2__class_3__garbage_truck.jpg"
    ) == ["class 1", "class 2", "class 3", "garbage truck"]
    choices = sample_mcq_choices(
        "/val/garbage_truck/0001.JPEG", pool, 3, seed=0
    )
    assert (
        len(choices) == 4
        and "garbage truck" in choices
        and len(set(choices)) == 4
    )
    assert choices == sample_mcq_choices(
        "/val/garbage_truck/0001.JPEG", pool, 3, seed=0
    )  # reproducible
    grid = sample_mcq_choices(
        "/g/grid_4_7__class_1__class_2__class_3__garbage_truck.jpg",
        pool,
        3,
        seed=0,
    )
    assert len(grid) == 7 and {
        "class 1",
        "class 2",
        "class 3",
        "garbage truck",
    } <= set(grid)
    with pytest.raises(ValueError, match="not among"):
        sample_mcq_choices("/val/zebra/1.JPEG", pool, 3, seed=0)
