"""Plots of the concept insertion/deletion curves (pipeline step 8).

plot_method   <eval>/<method>/concept_curves.png: mean insertion and deletion
              curves, concepts in activation order (solid) and random order
              (dashed), over the fraction of concepts
plot_summary  one bar per method: the gap and its random-order control
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from tgcl.evaluation.concept_insertion_deletion import CURVES
from tgcl.evaluation.concept_insertion_deletion import read_summary

INSERTION, DELETION = "#2a6fdb", "#d4492f"  # blue / red


def plot_method(method_dir: Path, title: str = "") -> Optional[Path]:
    """concept_curves.png in method_dir; None without concept_curves.npz."""
    method_dir = Path(method_dir)
    path = method_dir / CURVES
    if not path.exists():
        return None
    data = np.load(path)
    x = data["ms"] / float(data["k"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for key, color, label in (
        ("ins", INSERTION, "insertion"),
        ("del", DELETION, "deletion"),
    ):
        for order, style, name in (
            ("value", "-", "activation"),
            ("random", "--", "random"),
        ):
            y = data[f"{key}_{order}"].mean(axis=0)
            ax.plot(x, y, style, color=color, lw=2, label=f"{label} ({name})")
    summary = read_summary(method_dir) or {}
    gap = summary.get("gap")
    ax.set_xlabel("fraction of concepts")
    ax.set_ylabel("p(token) / p(full state)")
    ax.set_title(
        (title or method_dir.name)
        + (f"  gap {gap:.2f}" if gap is not None else ""),
        fontsize=10,
    )
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = method_dir / "concept_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_summary(
    eval_dir: Path, methods: List[str], out_png: Path
) -> Optional[Path]:
    """Bars of every method's gap and random-order gap; None without any
    summary."""
    rows = [
        (m, s)
        for m in methods
        if (s := read_summary(Path(eval_dir) / m)) is not None
    ]
    if not rows:
        return None
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(rows), 3.2))
    ax.bar(x - 0.18, [s["gap"] for _, s in rows], 0.34, color=INSERTION,
           label="activation order")  # fmt: skip
    ax.bar(x + 0.18, [s["gap_random"] for _, s in rows], 0.34,
           color="#9aa3ad", label="random order")  # fmt: skip
    ax.set_xticks(x, [m for m, _ in rows])
    ax.set_ylabel("insertion - deletion gap")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return Path(out_png)
