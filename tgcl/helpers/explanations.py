"""Compact storage of explainer outputs (vlm_explanations.json).

Every explained token lists its top concepts. The concept's grounding (class
names, logit-lens words, grounding image paths and box coordinates) is the same
wherever that concept appears, so it is stored once per file in
payload["concept_table"] (keyed by the concept index) instead of in every token
entry; segmentation masks are not stored at all (the concept bank .pth keeps
them). On ImageNet-1k this shrinks a SAM run's 50k-image file from ~7 GB to a
few hundred MB.

Readers call expand() (or load()), which puts the fields back into every entry
by reference (no copies), so code written for the old layout works unchanged;
old files without a table pass through as they are.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Union

TABLE_KEY = "concept_table"
GROUNDING_KEYS = (
    "concept_name",
    "text_grounding",
    "image_grounding_path",
    "image_grounding_bboxes",
)
MASK_KEY = "image_grounding_masks"


def concept_entries(payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Every top-concept entry: per token and over the whole sequence."""
    for result in payload.get("results", []):
        for token in result.get("per_token_concepts") or []:
            yield from token.get("top_concepts") or []
        yield from result.get("top_concepts_over_sequence") or []


def compact(
    payload: Dict[str, Any], keep_masks: bool = False
) -> Dict[str, Any]:
    """Move each concept's grounding into payload['concept_table'], in place."""
    table: Dict[str, Dict[str, Any]] = payload.setdefault(TABLE_KEY, {})
    keys = GROUNDING_KEYS + ((MASK_KEY,) if keep_masks else ())
    for entry in concept_entries(payload):
        if entry.get("concept_index") is None:
            continue
        idx = str(entry["concept_index"])
        grounding = {k: entry[k] for k in keys if k in entry}
        known = table.setdefault(idx, grounding)
        if known != grounding and grounding:
            raise ValueError(
                f"concept {idx}: different grounding in two entries"
            )
        for k in GROUNDING_KEYS + (MASK_KEY,):
            entry.pop(k, None)
    return payload


def expand(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fill every entry's grounding fields from the table (shared objects)."""
    table = payload.get(TABLE_KEY)
    if not table:
        return payload
    for entry in concept_entries(payload):
        row = table.get(str(entry.get("concept_index")))
        if row:
            for k, v in row.items():
                entry.setdefault(k, v)
    return payload


def load(path: Union[str, Path]) -> Dict[str, Any]:
    """vlm_explanations.json with every entry's grounding filled in."""
    with open(path, encoding="utf-8") as f:
        return expand(json.load(f))
