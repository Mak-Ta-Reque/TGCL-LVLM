#!/usr/bin/env python3
"""The paper's concept-quality table from scripts/dictionary_metrics.py CSVs:
method x decomposition rows, Sparsity / Instability / Overlap per dataset.

TGCL (SNMF) is the method itself: semi-NMF concepts combined by symmetric
alignment (Lowdin orthogonalization) and noise removal (keeping 50 % of each
concept's coordinates). Every other row (TGCL with SAE or SIMPLE, the
baselines) is reported on its concepts as learned
(tgcl/reporting/dictionary_metrics.py):

    Sparsity ↑      Gini sparsity of the concept vectors
    Instability ↓   1 - matched cosine between seeds
    Overlap ↓       L2 overlap (mean |cos|, coordinates below 1 % of the L2
                    norm dropped)

    D=outputs/dictionary_metrics
    python scripts/dictionary_table.py \\
        --csv ImageNet=$D/gemma3n_imagenet1k_concepts.csv \\
        --csv MSCOCO=$D/gemma3n_coco10_concepts.csv --out $D/table.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from tgcl.reporting import dictionary_metrics as dm

COLUMNS = [
    ("Sparsity↑", "gini_sparsity_mean", 2),
    ("Instability↓", "instability", 3),
    ("Overlap↓", "l2_overlap_mean", 2),
]


def load(path: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """CSV records keyed by (method, decomposition, post_processing)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {
            (r["method"], r["decomposition"], r["post_processing"]): r
            for r in csv.DictReader(fh)
        }


def table(tables: Dict[str, Dict]) -> Tuple[List[str], List[List[str]]]:
    """(header, rows) of the table; '–' where a record or value is missing.

    Args:
        tables: Dataset name -> load() of its CSV, in column order.

    Returns:
        The header and one row of cells per table row (dm.table_rows).
    """
    header = ["Method", "Decomp."] + [
        f"{name} {label}" for name in tables for label, _, _ in COLUMNS
    ]
    rows = []
    for method, key, label, post in dm.table_rows():
        cells = [method, label]
        for records in tables.values():
            rec = records.get((method, key, post)) or {}
            for _, col, digits in COLUMNS:
                value = rec.get(col, "")
                if value in ("", "nan"):  # missing, or one concept
                    cells.append("–")
                else:
                    cells.append(f"{float(value):.{digits}f}")
        rows.append(cells)
    return header, rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tables = {
        name: load(Path(path))
        for name, path in (entry.split("=", 1) for entry in args.csv)
    }
    header, rows = table(tables)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header] + rows)
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for cells in rows:
        print("| " + " | ".join(cells) + " |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
