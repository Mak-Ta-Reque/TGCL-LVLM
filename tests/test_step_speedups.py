"""Speed-ups of pipeline steps 4 and 5 must not change their outputs.

Step 4 (save_features.py) generates with the KV cache; the token-of-interest
hook must then extract the same hidden states as without it. Step 5
(analyse_features.py) decomposes tags in worker processes; each tag must get the
decomposition the sequential loop computes, whatever the order.
"""

import argparse
import os
import sys

import numpy as np
import pytest
import torch

import tgcl.helpers.utils as utils

# ---------------------------------------------------------------- step 4 ----


def _tiny_lm():
    from transformers import LlamaConfig
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = (
        None  # always generate max_new_tokens
    )
    return model


def _features(model, inputs, use_cache, token_id, new_tokens=8):
    """What save_features.inference + the token-of-interest hook store for one
    batch."""
    args = argparse.Namespace(
        token_of_interest="x",
        token_of_interest_idx=token_id,
        token_of_interest_start_token=0,
        save_only_generated_tokens=True,
        exact_match_modules_to_hook=True,
    )
    utils.clear_forward_hooks(model)
    utils.clear_hooks_variables()
    extract = utils.register_hooks(
        model,
        modules_to_hook=["model.norm"],
        hook_name="save_hidden_states_for_token_of_interest",
        args=args,
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )
    generated = [
        out[b, inputs["input_ids"].shape[1] :].tolist()
        for b in range(out.shape[0])
    ]
    result = extract(
        model_output=[out[b] for b in range(out.shape[0])],
        model_generated_output=generated,
        image=["a"] * out.shape[0],
    )
    utils.clear_forward_hooks(model)
    utils.clear_hooks_variables()
    return out, result


def test_kv_cache_gives_the_same_token_of_interest_states():
    model = _tiny_lm()
    # Left-padded batch of two prompts, as the VLM collate builds it.
    inputs = {
        "input_ids": torch.tensor([[0, 0, 5, 6, 7], [3, 4, 5, 6, 7]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
    }
    out_ref, _ = _features(model, inputs, use_cache=False, token_id=63)
    token_id = int(
        out_ref[0, 5 + 3]
    )  # the 4th generated token of row 0 is the "concept word"

    out_nc, ref = _features(model, inputs, use_cache=False, token_id=token_id)
    out_c, got = _features(model, inputs, use_cache=True, token_id=token_id)

    assert torch.equal(out_nc, out_c)  # same greedy answer
    mask_ref, mask_got = (
        ref["token_of_interest_mask"],
        got["token_of_interest_mask"],
    )
    assert mask_ref[0] and torch.equal(mask_ref, mask_got)
    for a, b in zip(
        ref["hidden_states"]["model.norm"], got["hidden_states"]["model.norm"]
    ):
        assert torch.allclose(a, b, atol=1e-5)


def test_feature_files_are_written_atomically(tmp_path):
    args = argparse.Namespace(
        save_dir=str(tmp_path),
        save_filename="results_cat",
        hook_names=["save_hidden_states_for_token_of_interest"],
    )
    paths = utils.feature_file_paths(args)
    assert paths == [
        str(
            tmp_path
            / "features"
            / "save_hidden_states_for_token_of_interest_results_cat.pth"
        )
    ]
    utils.save_hidden_states_to_file(
        {"hidden_states": [1]}, hook_name=args.hook_names[0], args=args
    )
    assert os.listdir(tmp_path / "features") == [
        os.path.basename(paths[0])
    ]  # no .tmp left
    assert torch.load(paths[0]) == {"hidden_states": [1]}


# ---------------------------------------------------------------- step 5 ----


def _write_tag(path, n, dim, seed):
    g = torch.Generator().manual_seed(seed)
    states = [torch.randn(1, dim, generator=g) for _ in range(n)]
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False  # one sample without the word: dropped before decomposing
    torch.save(
        {"hidden_states": {"norm": states}, "token_of_interest_mask": [mask]},
        path,
    )


def _tag_args(paths, method):
    return [
        argparse.Namespace(
            features_path=[str(p)],
            module_to_decompose="norm",
            decomposition_extract_pos=None,
            load_matched_features=False,
            num_concepts=[2],
            decomposition_method=method,
            dl_alpha=1.0,
            dl_max_iter=50,
            seed=1,
        )
        for p in paths
    ]


@pytest.mark.parametrize("method", ["snmf", "pca"])
def test_parallel_decomposition_equals_sequential_in_any_order(
    tmp_path, method
):
    import tgcl.analyse_features as af
    from tgcl.analysis import decompose_feature_file

    paths = [tmp_path / f"results_{tag}.pth" for tag in ("cat", "dog", "car")]
    for i, p in enumerate(paths):
        _write_tag(p, n=12, dim=40, seed=i)
    tag_args = _tag_args(paths, method)

    sequential = {
        a.features_path[0]: decompose_feature_file(a) for a in tag_args
    }
    reversed_order = {
        a.features_path[0]: decompose_feature_file(a)
        for a in reversed(tag_args)
    }
    parallel = af.precompute_decompositions(tag_args, workers=2)

    assert set(parallel) == set(sequential)
    for key, (concepts, activations, _) in sequential.items():
        assert concepts.shape == (2, 40) and activations.shape == (
            11,
            2,
        )  # masked sample dropped
        for other in (reversed_order, parallel):
            np.testing.assert_array_equal(
                np.asarray(other[key][0]), np.asarray(concepts)
            )
            np.testing.assert_array_equal(
                np.asarray(other[key][1]), np.asarray(activations)
            )


def test_parallel_decomposition_skips_gpu_methods_and_single_tags(tmp_path):
    import tgcl.analyse_features as af

    p = tmp_path / "results_cat.pth"
    _write_tag(p, n=12, dim=8, seed=0)
    assert (
        af.precompute_decompositions(_tag_args([p, p], "sae2"), workers=4) == {}
    )
    assert af.precompute_decompositions(_tag_args([p], "snmf"), workers=4) == {}
    assert (
        af.precompute_decompositions(_tag_args([p, p], "snmf"), workers=1) == {}
    )


def test_decomp_workers_env(monkeypatch):
    import tgcl.analyse_features as af

    monkeypatch.setenv("DECOMP_WORKERS", "3")
    assert af.decomp_workers() == 3
    monkeypatch.delenv("DECOMP_WORKERS")
    assert 1 <= af.decomp_workers() <= 8


# ---------------------------------------------------------------- step 6 ----


def test_explainer_cosine_matmul_equals_broadcast_cosine():
    from tgcl.inference.vlm_explainer_multibatch import cosine_to_concepts

    g = torch.Generator().manual_seed(0)
    acts = torch.randn(5, 64, generator=g).to(
        torch.bfloat16
    )  # hidden states as captured (bf16)
    concepts = torch.randn(300, 64, generator=g)
    unit = torch.nn.functional.normalize(concepts, dim=1, eps=1e-8)
    # Exact float32 cosine (the old broadcast call with bf16 tokens rounded in
    # bf16, up to ~3e-3).
    ref = torch.nn.functional.cosine_similarity(
        acts.float().unsqueeze(1), concepts.unsqueeze(0), dim=2
    )
    got = cosine_to_concepts(acts, unit)
    assert got.shape == (5, 300) and torch.allclose(got, ref, atol=1e-5)
    assert torch.equal(torch.topk(got, 3).indices, torch.topk(ref, 3).indices)


def test_stored_token_embedding_does_not_pin_the_sequence():
    seq = torch.randn(
        300, 64
    )  # float32 already: .to() returns the same storage
    row = seq[-2].detach().to(dtype=torch.float32).cpu().numpy().copy()
    assert row.base is None and not np.shares_memory(row, seq.numpy())


def test_explainer_step_frees_large_allocations(monkeypatch):
    import importlib.util as iu
    import types

    sys.modules.setdefault(
        "dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
    )
    spec = iu.spec_from_file_location(
        "rfp",
        os.path.join(
            os.path.dirname(__file__), "..", "scripts", "run_full_pipeline.py"
        ),
    )
    src = open(spec.origin, encoding="utf-8").read()
    assert (
        'env_overrides={"MALLOC_MMAP_THRESHOLD_": "131072"}'
        in src.split("def step_6_vlm_explainer")[1].split("\ndef ")[0]
    )


def test_segment_image_is_cached_and_identical(tmp_path):
    from PIL import Image

    from tgcl.helpers import segment_view as sv

    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(
        tmp_path / "a.png"
    )
    path = str(tmp_path / "a.png")
    old = (
        Image.open(path)
        .convert("RGB")
        .resize((512, 384), resample=Image.Resampling.LANCZOS)
    )
    first = sv.segment_image(path, [512, 384])
    assert np.array_equal(np.asarray(first), np.asarray(old))
    assert sv.segment_image(path, [512, 384]) is first  # decoded once
    assert sv.segment_image(path, None).size == (
        640,
        480,
    )  # no size: the image as stored
