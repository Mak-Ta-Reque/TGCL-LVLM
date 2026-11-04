"""Directory layout of one pipeline run (scripts/run_full_pipeline.py).

A run directory holds features, concept banks and, per evaluation variant,
explanations and evaluation results:

    <run>/concept/<method>/combined_concept_<method>_<ratio>_raw.pth
    <run>/explanations[_<eval_tag>]/<method>/vlm_explanations.json
    <run>/eval[_<eval_tag>]/<method>/...

`eval_tag` separates several evaluations of the same concepts (single-object vs
grid images, text-only / image-only / combined retrieval). An empty tag is the
default layout. This module is the single definition used by the pipeline and by
every evaluation/report script.
"""

from pathlib import Path
import re
from typing import Optional

_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_eval_tag(eval_tag: str) -> str:
    """Return `eval_tag` stripped; raise ValueError if it is not a safe name."""
    tag = (eval_tag or "").strip()
    if tag and not _TAG_PATTERN.match(tag):
        raise ValueError(
            f"EVAL_TAG must be letters, digits, '_' or '-' (got {eval_tag!r})"
        )
    return tag


def _tagged(name: str, eval_tag: str) -> str:
    tag = validate_eval_tag(eval_tag)
    return f"{name}_{tag}" if tag else name


def explanations_dir(run_dir: Path, eval_tag: str = "") -> Path:
    return Path(run_dir) / _tagged("explanations", eval_tag)


def eval_dir(run_dir: Path, eval_tag: str = "") -> Path:
    return Path(run_dir) / _tagged("eval", eval_tag)


def plots_dir(run_dir: Path, eval_tag: str = "") -> Path:
    return Path(run_dir) / _tagged("plots", eval_tag)


def explanations_json(run_dir: Path, method: str, eval_tag: str = "") -> Path:
    return (
        explanations_dir(run_dir, eval_tag) / method / "vlm_explanations.json"
    )


def find_concept_bank(run_dir: Path, method: str) -> Optional[Path]:
    """The run's combined (positive) concept bank for `method`, or None."""
    banks = sorted(
        p
        for p in (Path(run_dir) / "concept" / method).glob(
            f"combined_concept_{method}_*_raw.pth"
        )
        if "negative" not in p.name
    )
    if len(banks) > 1:
        raise ValueError(
            f"Several concept banks for {method} in {run_dir}: "
            f"{[b.name for b in banks]}"
        )
    return banks[0] if banks else None
