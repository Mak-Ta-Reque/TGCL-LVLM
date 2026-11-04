#!/usr/bin/env python3
"""Join class-sharded pipeline runs (CONCEPT_SHARD=k/K, --stop-after-step 5)
into one run.

For every decomposition method, every concept-bank file the shards share
(<shard>/concept/<method>/*.pth: raw, gl, negative and regrounded banks) is
concatenated in shard order: tensors along dim 0, lists end to end; any other
value (method name, clean-example ratio, ...) must be equal in all shards.
Per-tag feature files are symlinked into <out>/features/. Then run steps 6-8
on the merged run:

    python scripts/merge_concept_shards.py \\
        --shards outputs/paper/x_shard0of4 ... --out outputs/paper/x_seed1
    python scripts/run_full_pipeline.py --output-dir outputs/paper/x_seed1 \\
        --skip-to-step 6
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch


def merge_banks(banks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate the shards' versions of one concept bank (see module doc).

    Raises:
        ValueError: The banks' keys or shared values differ.
    """
    keys = list(banks[0])
    for i, bank in enumerate(banks[1:], 1):
        if list(bank) != keys:
            raise ValueError(
                f"shard {i} bank keys {list(bank)} differ from {keys}"
            )
    merged: Dict[str, Any] = {}
    for key in keys:
        values = [bank[key] for bank in banks]
        if all(torch.is_tensor(v) for v in values):
            merged[key] = torch.cat(values, dim=0)
        elif all(isinstance(v, list) for v in values):
            merged[key] = [x for v in values for x in v]
        elif all(v == values[0] for v in values):
            merged[key] = values[0]
        elif key == "sparsification" and all(
            isinstance(v, dict) for v in values
        ):
            # CONCEPT_SPARSIFY runs per shard: the setting must agree, the
            # fidelity
            # statistics (measured on each shard's states) are kept per shard.
            setting = {
                k: values[0].get(k) for k in ("algorithm", "density", "n_keep")
            }
            if any({k: v.get(k) for k in setting} != setting for v in values):
                raise ValueError(
                    f"'sparsification' setting differs between shards: {values}"
                )
            merged[key] = {**setting, "per_shard": values}
        else:
            raise ValueError(f"'{key}' differs between shards: {values}")
    return merged


def merge_runs(shards: List[Path], out: Path, methods: List[str]) -> List[Path]:
    """Merge every concept bank of every method into out and symlink the shards'
    feature files into out/features.

    Returns:
        The merged bank files.

    Raises:
        FileNotFoundError: A bank is missing in some shard.
        FileExistsError: Two shards have the same feature file.
    """
    written = []
    for method in methods:
        names = sorted(
            {p.name for p in (shards[0] / "concept" / method).glob("*.pth")}
        )
        if not names:
            raise FileNotFoundError(
                f"no concept banks in {shards[0] / 'concept' / method}"
            )
        for name in names:
            paths = [shard / "concept" / method / name for shard in shards]
            missing = [str(p) for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{name} missing in: {missing}")
            banks = [
                torch.load(p, map_location="cpu", weights_only=False)
                for p in paths
            ]
            target = out / "concept" / method / name
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(merge_banks(banks), target)
            written.append(target)
    features = out / "features"
    features.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        for path in sorted((shard / "features").glob("*.pth")):
            link = features / path.name
            if link.exists() or link.is_symlink():
                raise FileExistsError(
                    f"feature file {path.name} exists in two shards"
                )
            link.symlink_to(path.resolve())
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--shards",
        nargs="+",
        type=Path,
        required=True,
        help="shard run directories, in shard order",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="merged run directory"
    )
    ap.add_argument("--methods", default="snmf")
    args = ap.parse_args()
    written = merge_runs(
        args.shards,
        args.out,
        [m.strip() for m in args.methods.split(",") if m.strip()],
    )
    for path in written:
        bank = torch.load(path, map_location="cpu", weights_only=False)
        n = (
            bank["concepts"].shape[0]
            if torch.is_tensor(bank.get("concepts"))
            else "?"
        )
        print(f"{path}: {n} concepts")


if __name__ == "__main__":
    main()
