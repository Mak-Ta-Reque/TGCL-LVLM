"""tgcl.models.revisions: every model the presets use is pinned to a commit."""

from pathlib import Path
import re

from tgcl.models.revisions import MODEL_REVISIONS
from tgcl.models.revisions import revision

REPO = Path(__file__).resolve().parent.parent


def test_every_preset_model_is_pinned():
    models = set()
    for env in (REPO / "configs").rglob("*.env"):
        models |= set(
            re.findall(r"^(?:export )?VLM_MODEL=(\S+)", env.read_text(), re.M)
        )
    assert models, "no VLM_MODEL in the presets"
    assert models <= set(MODEL_REVISIONS), models - set(MODEL_REVISIONS)


def test_revisions_are_full_commit_hashes():
    for name, commit in MODEL_REVISIONS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", commit), name


def test_unpinned_names_and_local_paths_get_no_revision():
    assert revision("/models/local-checkpoint") is None
    assert revision("google/gemma-3n-E4B-it") == (
        MODEL_REVISIONS["google/gemma-3n-E4B-it"]
    )
