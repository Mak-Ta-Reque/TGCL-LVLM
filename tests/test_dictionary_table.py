"""scripts/dictionary_table.py and the paper's row layout: TGCL's SNMF with its
combination (symmetric alignment + noise removal, keep 50 %), every other row
as learned."""

import csv

import dictionary_table as dt

from tgcl.reporting import dictionary_metrics as dm

(KEEP50,) = dm.TGCL_COMBINATIONS
FIELDS = [
    "method",
    "decomposition",
    "post_processing",
    "gini_sparsity_mean",
    "instability",
    "l2_overlap_mean",
]
TOY = [  # method, decomposition, post, gini, instability, l2 overlap
    ("TGCL (SAM)", "snmf", "none", 0.44, 0.0, 0.72),
    ("TGCL (SAM)", "snmf", KEEP50, 0.61, 0.004, 0.01),
    ("TGCL (SAM)", "simple", "none", 0.45, 0.09, 0.30),
    ("TGCL (SAM)", "simple", KEEP50, 0.61, 0.05, 0.02),
    ("CoX-LMM", "snmf", "none", 0.43, 0.002, 0.32),
    ("CoX-LMM", "snmf", KEEP50, 0.61, 0.02, 0.02),
    ("CRAFT", "simple", "none", 0.42, 0.33, "nan"),
]


def _records(tmp_path):
    path = tmp_path / "toy.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in TOY:
            w.writerow(dict(zip(FIELDS, r)))
    return dt.load(path)


def _row(rows, method, label):
    return next(r for r in rows if r[0] == method and r[1] == label)


def test_row_layout():
    assert KEEP50 == "lowdin+magnitude:0.5"
    rows = dm.table_rows()
    tgcl = [r for r in rows if r[0] == "TGCL (SAM)"]
    assert [(k, post) for _, k, _, post in tgcl] == [
        ("snmf", KEEP50),
        ("sae2", "none"),
        ("simple", "none"),
    ]
    assert all(post == "none" for m, _, _, post in rows if not dm.is_tgcl(m))
    assert len(rows) == 5 * 3


def test_tgcl_snmf_combined_every_other_row_as_learned(tmp_path):
    header, rows = dt.table({"MSCOCO": _records(tmp_path)})
    assert header[2:] == [
        "MSCOCO Sparsity↑",
        "MSCOCO Instability↓",
        "MSCOCO Overlap↓",
    ]
    assert _row(rows, "TGCL (SAM)", "SNMF")[2:] == ["0.61", "0.004", "0.01"]
    assert _row(rows, "TGCL (SAM)", "SIMPLE")[2:] == ["0.45", "0.090", "0.30"]
    assert _row(rows, "CoX-LMM", "SNMF")[2:] == ["0.43", "0.002", "0.32"]
    assert _row(rows, "CRAFT", "SIMPLE")[2:] == ["0.42", "0.330", "–"]
    assert _row(rows, "MCD", "SNMF")[2:] == ["–", "–", "–"]
