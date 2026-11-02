"""
Interpretability figures for prototype networks (requires matplotlib).

Ported from ``show_anchor_grid``, ``plot_umap_by_prototype`` and
``plot_proto_errors`` (``evaluation_tools/pnet_support.py``).

Porting changes vs. source
--------------------------
- Figures are returned instead of ``plt.show()``-ed, so they can be saved.
- ``plot_embedding_by_prototype`` takes any 2-D projection (UMAP, t-SNE, PCA);
  computing it is left to the caller, so ``umap-learn`` is not a dependency.
- ``plot_prototype_errors`` takes the table of
  :func:`~tomographia.prototypes.per_prototype_confusion` (class of each
  prototype no longer hard-coded).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib: pip install 'tomographia[report]'") from exc
    return plt


def plot_prototype_examples(paths_by_proto: Dict[int, Sequence[str]], n_cols: int = 5,
                            labels: Optional[Dict[int, str]] = None, cmap: str = "gray",
                            figsize=None):
    """
    Grid of example patches: one row per prototype, nearest examples first.

    ``paths_by_proto`` maps prototype id to image paths, e.g.
    ``{pid: df.path.iloc[rows] for pid, rows in nearest_examples(...).items()}``.
    """
    plt = _plt()
    from ..patches.dataset import load_patch

    pids = list(paths_by_proto)
    fig, axs = plt.subplots(len(pids), n_cols, squeeze=False,
                            figsize=figsize or (2 * n_cols, 2 * len(pids)))
    for r, pid in enumerate(pids):
        paths = list(paths_by_proto[pid])[:n_cols]
        for c in range(n_cols):
            ax = axs[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(paths):
                ax.imshow(load_patch(paths[c]), cmap=cmap)
            else:
                ax.axis("off")
        axs[r, 0].set_ylabel(labels.get(pid, f"proto {pid}") if labels else f"proto {pid}")
    fig.tight_layout()
    return fig


def plot_embedding_by_prototype(coords: np.ndarray, assignment: Sequence[int],
                                proto_coords: Optional[np.ndarray] = None,
                                labels: Optional[Dict[int, str]] = None,
                                point_size: float = 6, cmap: str = "tab10", ax=None):
    """
    Scatter a 2-D projection of the embeddings, coloured by nearest prototype.

    ``proto_coords`` (``[K, 2]``) are drawn as labelled squares.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    assignment = np.asarray(assignment)
    k = int(max(assignment.max() + 1, 0 if proto_coords is None else len(proto_coords)))
    colors = plt.get_cmap(cmap, max(k, 1))
    for pid in range(k):
        m = assignment == pid
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=point_size, color=colors(pid),
                       alpha=0.6, linewidths=0,
                       label=labels.get(pid, f"proto {pid}") if labels else f"proto {pid}")
    if proto_coords is not None:
        ax.scatter(proto_coords[:, 0], proto_coords[:, 1], s=160, marker="s",
                   c=[colors(i) for i in range(len(proto_coords))], edgecolor="k")
        for pid, (x, y) in enumerate(proto_coords):
            ax.text(x, y, str(pid), ha="center", va="center", fontsize=8, color="white")
    ax.legend(loc="best", markerscale=2, frameon=True)
    return ax.figure


def plot_prototype_errors(confusion, ax=None):
    """Stacked FN / FP bars per prototype, split by prototype class."""
    plt = _plt()
    classes = sorted(confusion["proto_class"].unique())
    if ax is None:
        _, ax = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4),
                             sharey=True, squeeze=False)
        axs = ax[0]
    else:
        axs = np.atleast_1d(ax)
    for a, c in zip(axs, classes):
        block = confusion[confusion.proto_class == c]
        x = [f"p{p}" for p in block.proto]
        a.bar(x, block.fn, color="#E15759", label="FN")
        a.bar(x, block.fp, bottom=block.fn, color="#F28E2B", label="FP")
        a.set_title(f"Class {c} prototypes")
        a.set_xlabel("prototype")
        a.legend(frameon=False)
    axs[0].set_ylabel("count")
    axs[0].figure.tight_layout()
    return axs[0].figure
