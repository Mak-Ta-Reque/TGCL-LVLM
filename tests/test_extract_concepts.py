"""scripts/extract_concepts.py: the job graph that reproduces the released
concepts from the beginning."""

import extract_concepts as ec


def _by_name(jobs):
    return {j.name: j for j in jobs}


def test_concept_jobs_coco10(tmp_path):
    jobs = _by_name(ec.concept_jobs("coco10", tmp_path, seeds=(1, 2)))
    assert len(jobs) == 2 * 5
    sam = jobs["coco10_tgcl_sam_s2"]
    assert (
        sam.env["SEED"] == "2"
        and sam.env["DECOMP_METHODS"] == "snmf,simple,sae2"
    )
    assert sam.env["CONCEPT_ORTHO"] == "lowdin"  # TGCL's combination preset
    assert sam.cmd[-2:] == ["--stop-after-step", "5"]
    assert sam.cmd[sam.cmd.index("--output-dir") + 1] == str(
        tmp_path / "concepts" / "coco10" / "tgcl_sam_seed2"
    )
    mcd = jobs["coco10_mcd_s2"]
    assert mcd.needs == {"coco10_tgcl_sam_s2"} and mcd.copy_crops
    assert mcd.copy_from == tmp_path / "concepts" / "coco10" / "tgcl_sam_seed2"
    assert mcd.env["DECOMP_COMPONENTS_GLOBAL"] == "10"
    assert mcd.env["DECOMP_METHODS"] == "snmf,simple,sae2"  # not MCD's PCA
    assert not jobs["coco10_craft_s1"].copy_crops


def test_state_jobs(tmp_path):
    coco = ec.state_jobs("coco10", tmp_path, seeds=(1, 2))
    assert len(coco) == 2 * 4  # four settings per seed
    first = coco[0]
    assert first.needs == {"coco10_tgcl_sam_s1"} and first.evaluation
    assert first.cmd[first.cmd.index("--output-dir") + 1] == str(
        tmp_path / "test_states" / "coco10" / "seed1"
    )
    assert {j.env["SEED"] for j in coco} == {"1", "2"}
    imagenet = ec.state_jobs("imagenet1k", tmp_path)
    assert len(imagenet) == 4 and {j.env["SEED"] for j in imagenet} == {"1"}
    assert len({j.name for j in coco + imagenet}) == 12
