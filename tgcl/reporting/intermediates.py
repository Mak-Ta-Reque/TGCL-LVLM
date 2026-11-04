"""Editable intermediate outputs between the pipeline and the paper metrics.

`export_run` reads a finished pipeline run and writes every metric's input as a
plain file under <run>/intermediate/ (see README_TEXT for the formats).
scripts/paper_report.py computes all final metrics from these files only, so
an intermediate can be edited (reshaped, filtered, re-labelled) and the
metrics re-run without the model. Existing files are never overwritten unless
overwrite=True, so edits survive re-exports.
"""

from collections import Counter
import csv
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional

import numpy as np
import torch

from tgcl.evaluation.concept_insertion_deletion import (
    SUMMARY as CONCEPT_CURVES_SUMMARY,
)
from tgcl.helpers.run_layout import eval_dir
from tgcl.helpers.run_layout import explanations_json
from tgcl.helpers.run_layout import find_concept_bank
from tgcl.helpers.run_layout import validate_eval_tag
from tgcl.reporting.report_metrics import load_concepts
from tgcl.reporting.report_metrics import load_predictions_from_json

RETRIEVAL_TAGS = ("", "text_only", "image_only", "combined")

README_TEXT = """# Intermediate outputs (input of scripts/paper_report.py)

Every final metric is computed from these files only. Edit them (keep the
format) and re-run `python scripts/paper_report.py tables ...` or `ablation ...`.
`python scripts/paper_report.py export --runs ...` fills in missing files and
never overwrites existing ones (pass --overwrite to regenerate).

token_hits.csv                       one row per step-4 sample: concept, found (0/1)
                                     -> token_hit_rate.csv
<method>/concepts.npy                (K, D) float32 concept matrix, one row per concept
                                     -> sparsity, near-zero fraction, overlap, instability
<method>/object_predictions[_<tag>].csv
                                     image, true_label, pred1, pred2 (canonical class names)
                                     -> object_f1.csv
<method>/retrieval[_<tag>].csv       sample, image_path, rank_k, bert, clip, random_bert, random_clip
                                     (sample = ALL: only the run's mean was available)
                                     -> retrieval.csv
<method>/word_alignment[_<tag>].csv
                                     concept-word alignment per test image: top_concepts and
                                     {word_clipscore,random_word_clipscore,phrase_bertscore,random_phrase_bertscore}_top{j}
                                     (score of the j-th concept by activation)
                                     -> word_alignment.csv (top-k = mean of top1..topk)
<method>/concept_curves[_<tag>].json concept insertion/deletion summary
                                     (tgcl/evaluation/concept_insertion_deletion.py): gap,
                                     gap_random, gap_net, insertion_auc, deletion_auc, n_images
                                     -> faithfulness.csv

<tag> is the evaluation variant (EVAL_TAG); no suffix = the base evaluation.
"""


def intermediate_dir(run: Path) -> Path:
    return Path(run) / "intermediate"


def _suffixed(name: str, ext: str, eval_tag: str) -> str:
    tag = validate_eval_tag(eval_tag)
    return f"{name}_{tag}{ext}" if tag else f"{name}{ext}"


def concepts_path(run: Path, method: str) -> Path:
    return intermediate_dir(run) / method / "concepts.npy"


def token_hits_path(run: Path) -> Path:
    return intermediate_dir(run) / "token_hits.csv"


def object_predictions_path(run: Path, method: str, eval_tag: str = "") -> Path:
    return (
        intermediate_dir(run)
        / method
        / _suffixed("object_predictions", ".csv", eval_tag)
    )


def retrieval_path(run: Path, method: str, eval_tag: str = "") -> Path:
    return (
        intermediate_dir(run)
        / method
        / _suffixed("retrieval", ".csv", eval_tag)
    )


def word_alignment_path(run: Path, method: str, eval_tag: str = "") -> Path:
    return (
        intermediate_dir(run)
        / method
        / _suffixed("word_alignment", ".csv", eval_tag)
    )


def concept_curves_path(run: Path, method: str, eval_tag: str = "") -> Path:
    return (
        intermediate_dir(run)
        / method
        / _suffixed("concept_curves", ".json", eval_tag)
    )


# ---------------------------------------------------------------- writers


def _write_rows(path: Path, header: List[str], rows: List[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _export_concepts(run: Path, method: str, path: Path) -> bool:
    bank = find_concept_bank(run, method)
    concepts = load_concepts(bank) if bank else None
    if concepts is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, concepts.numpy().astype(np.float32))
    return True


def _export_token_hits(run: Path, path: Path) -> bool:
    rows = []
    for feature_file in sorted((Path(run) / "features").glob("*.pth")):
        data = torch.load(feature_file, map_location="cpu", weights_only=False)
        masks = data.get("token_of_interest_mask")
        if masks is None:
            continue
        found = (
            torch.cat([torch.as_tensor(m).reshape(-1) for m in masks])
            .bool()
            .tolist()
        )
        names = data.get("concept") or []
        default = (
            Counter(names).most_common(1)[0][0] if names else feature_file.stem
        )
        for i, hit in enumerate(found):
            rows.append([names[i] if i < len(names) else default, int(hit)])
    if not rows:
        return False
    _write_rows(path, ["concept", "found"], rows)
    return True


def _export_object_predictions(
    run: Path, method: str, eval_tag: str, path: Path
) -> bool:
    json_path = explanations_json(run, method, eval_tag)
    rows = load_predictions_from_json(json_path)
    if not rows:
        return False
    with open(json_path, encoding="utf-8") as f:
        images = [
            r.get("image_path", "") for r in json.load(f).get("results", [])
        ]
    _write_rows(
        path,
        ["image", "true_label", "pred1", "pred2"],
        [
            [img, t, p1 or "", p2 or ""]
            for img, (t, p1, p2) in zip(images, rows)
        ],
    )
    return True


def _export_retrieval(
    run: Path, method: str, eval_tag: str, path: Path
) -> bool:
    folder = eval_dir(run, eval_tag) / method
    per_sample = folder / "clip_bert_topk_per_sample.csv"
    if per_sample.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(per_sample, path)
        return True
    summary = folder / "clip_bert_topk_table.csv"
    if not summary.exists():
        return False
    with open(summary, newline="", encoding="utf-8") as f:
        rows = [
            [
                "ALL",
                "",
                r["rank_k"],
                r["bert_mean"],
                r["clip_mean"],
                r["random_bert_mean"],
                r["random_clip_mean"],
            ]
            for r in csv.DictReader(f)
        ]
    _write_rows(
        path,
        [
            "sample",
            "image_path",
            "rank_k",
            "bert",
            "clip",
            "random_bert",
            "random_clip",
        ],
        rows,
    )
    return True


def _export_word_alignment(
    run: Path, method: str, eval_tag: str, path: Path
) -> bool:
    source = eval_dir(run, eval_tag) / method / "word_alignment_per_sample.csv"
    if not source.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    return True


def _export_concept_curves(
    run: Path, method: str, eval_tag: str, path: Path
) -> bool:
    source = eval_dir(run, eval_tag) / method / CONCEPT_CURVES_SUMMARY
    if not source.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    return True


def export_run(
    run: Path, methods: List[str], eval_tags: List[str], overwrite: bool = False
) -> List[Path]:
    """Write the missing intermediates of one run; returns the files written."""
    run = Path(run)
    written: List[Path] = []

    def do(path: Path, export, *args) -> None:
        """export(*args, path) unless path exists (and not overwrite)."""
        if path.exists() and not overwrite:
            return
        if export(*args, path):
            written.append(path)

    do(token_hits_path(run), _export_token_hits, run)
    for method in methods:
        do(concepts_path(run, method), _export_concepts, run, method)
        for tag in sorted(set(eval_tags) | set(RETRIEVAL_TAGS)):
            args = (run, method, tag)
            do(
                object_predictions_path(*args),
                _export_object_predictions,
                *args,
            )
            do(retrieval_path(*args), _export_retrieval, *args)
            do(concept_curves_path(*args), _export_concept_curves, *args)
            do(word_alignment_path(*args), _export_word_alignment, *args)
    if written:
        readme = intermediate_dir(run) / "README.md"
        readme.write_text(README_TEXT)
    return written


# ---------------------------------------------------------------- readers


def read_concepts(run: Path, method: str) -> Optional[torch.Tensor]:
    path = concepts_path(run, method)
    if not path.exists():
        return None
    concepts = np.load(path)
    if concepts.ndim != 2 or concepts.size == 0:
        return None
    return torch.from_numpy(np.ascontiguousarray(concepts, dtype=np.float32))


def read_token_hits(run: Path) -> List[tuple]:
    """[(concept, found: bool), ...]"""
    path = token_hits_path(run)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [
            (r["concept"], r["found"] in ("1", "True", "true"))
            for r in csv.DictReader(f)
        ]


def read_object_predictions(
    run: Path, method: str, eval_tag: str = ""
) -> List[tuple]:
    """[(true_label, pred1 or None, pred2 or None), ...]"""
    path = object_predictions_path(run, method, eval_tag)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [
            (r["true_label"], r["pred1"] or None, r["pred2"] or None)
            for r in csv.DictReader(f)
        ]


def read_retrieval(
    run: Path, method: str, eval_tag: str = ""
) -> Dict[int, Dict[str, float]]:
    """{rank_k: {"bert", "clip", "random_bert", "random_clip"}} averaged over
    samples."""
    path = retrieval_path(run, method, eval_tag)
    if not path.exists():
        return {}
    sums: Dict[int, Dict[str, List[float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per_rank = sums.setdefault(
                int(r["rank_k"]),
                {k: [] for k in ("bert", "clip", "random_bert", "random_clip")},
            )
            for key in per_rank:
                if r.get(key) not in (None, ""):
                    per_rank[key].append(float(r[key]))
    return {
        k: {m: float(np.mean(v)) for m, v in d.items() if v}
        for k, d in sums.items()
    }


def read_word_alignment(
    run: Path, method: str, eval_tag: str = ""
) -> Dict[str, Dict[int, float]]:
    """{metric: {k: mean over samples of the mean of the top-1..k concept
    scores}}."""
    path = word_alignment_path(run, method, eval_tag)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: Dict[str, Dict[int, float]] = {}
    for col in rows[0]:
        if "_top" not in col:
            continue
        metric, j = col.rsplit("_top", 1)
        out.setdefault(metric, {})
        out[metric][int(j)] = None  # placeholder: filled below per k
    for metric, ks in out.items():
        for k in sorted(ks):
            per_sample = []
            for r in rows:
                vals = [
                    float(r[f"{metric}_top{j}"])
                    for j in range(1, k + 1)
                    if r.get(f"{metric}_top{j}") not in (None, "")
                ]
                if vals:
                    per_sample.append(float(np.mean(vals)))
            ks[k] = float(np.mean(per_sample)) if per_sample else None
    return out


def read_concept_curves(
    run: Path, method: str, eval_tag: str = ""
) -> Optional[Dict[str, float]]:
    """The concept insertion/deletion summary, or None."""
    path = concept_curves_path(run, method, eval_tag)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
