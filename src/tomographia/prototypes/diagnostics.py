"""
Prototype diagnostics and semantic metrics.

Anchor assignment accuracy, prototype drift, learned-anchor purity,
coalescence and per-prototype confusion.  Functions take numpy arrays (or a
head) so they work on the embeddings returned by
:func:`tomographia.prototypes.predict_prototype_network`.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model import PrototypeHead


def prototype_classes(k_per_class: Sequence[int]) -> np.ndarray:
    """Class id of each prototype, e.g. ``[3, 3] -> [0, 0, 0, 1, 1, 1]``."""
    return np.repeat(np.arange(len(k_per_class)), k_per_class)


@torch.no_grad()
def coalescence_stats(head: PrototypeHead, thresholds: Sequence[float] = (0.05, 0.10, 0.20)
                      ) -> Dict[int, Dict]:
    """
    Within-class distances between L2-normalised prototypes (collapse check).

    Returns ``{class: {"k", "min", "mean", "median", "closest_pair",
    "count<=thr"...}}``.
    """
    P = F.normalize(head.prototypes.detach(), dim=1)
    PD = torch.cdist(P, P)
    stats = {}
    for c, s in enumerate(head.class_slices):
        block = PD[s, s]
        k = block.shape[0]
        if k < 2:
            stats[c] = {"k": k}
            continue
        iu = torch.triu_indices(k, k, offset=1)
        vals = block[iu[0], iu[1]]
        j = int(torch.argmin(vals))
        st = {"k": k, "min": float(vals.min()), "mean": float(vals.mean()),
              "median": float(vals.median()),
              "closest_pair": (int(iu[0, j]), int(iu[1, j]))}
        for t in thresholds:
            st[f"count<={t:.2f}"] = int((vals <= t).sum())
        stats[c] = st
    return stats


@torch.no_grad()
def prototype_drift(head: PrototypeHead, reference: np.ndarray) -> np.ndarray:
    """L2 distance of each prototype from a reference (e.g. its initial medoid)."""
    ref = torch.as_tensor(np.asarray(reference), dtype=torch.float32, device=head.prototypes.device)
    return (head.prototypes.detach() - ref).norm(dim=1).cpu().numpy()


def standardize(head: PrototypeHead, embeddings: np.ndarray) -> np.ndarray:
    """Standardise raw embeddings with the head's scaler."""
    mean = head.scaler_mean.cpu().numpy()
    scale = head.scaler_scale.cpu().numpy()
    return (np.asarray(embeddings) - mean) / scale


def _sq_dists(z: np.ndarray, P: np.ndarray) -> np.ndarray:
    """``[N, K]`` squared distances without materialising ``[N, K, D]``."""
    z = np.asarray(z, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    d = (z ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :] - 2.0 * z @ P.T
    return np.maximum(d, 0.0)


def assign_prototypes(head: PrototypeHead, embeddings: np.ndarray) -> np.ndarray:
    """Index of the nearest prototype for each raw embedding."""
    z = standardize(head, embeddings)
    P = head.prototypes.detach().cpu().numpy()
    d = _sq_dists(z, P)
    return d.argmin(1)


def anchor_assignment_accuracy(head: PrototypeHead, anchor_embeddings: np.ndarray,
                               anchor_class: Sequence[int], anchor_proto: Sequence[int]) -> float:
    """Fraction of expert anchors whose nearest prototype is their designated one."""
    offsets = np.concatenate([[0], np.cumsum(head.k_per_class)[:-1]])
    target = offsets[np.asarray(anchor_class)] + np.asarray(anchor_proto)
    return float((assign_prototypes(head, anchor_embeddings) == target).mean())


def nearest_examples(head: PrototypeHead, embeddings: np.ndarray, k: int = 20,
                     restrict_to_class: Optional[Sequence[int]] = None) -> Dict[int, np.ndarray]:
    """
    "Learned anchors": the *k* samples closest to each prototype.

    Parameters
    ----------
    embeddings        : ``[N, D]`` raw embeddings (e.g. the training set).
    restrict_to_class : optional ``[N]`` labels; only samples of the
                        prototype's class are considered.

    Returns ``{prototype id: row indices into embeddings, nearest first}``.
    """
    z = standardize(head, embeddings)
    P = head.prototypes.detach().cpu().numpy()
    d = _sq_dists(z, P)
    pcls = prototype_classes(head.k_per_class)
    out = {}
    for pid in range(P.shape[0]):
        rows = np.arange(len(z))
        if restrict_to_class is not None:
            rows = rows[np.asarray(restrict_to_class) == pcls[pid]]
        out[pid] = rows[np.argsort(d[rows, pid])[:k]]
    return out


def learned_anchor_purity(neighbours: Dict[int, np.ndarray], sample_values: Sequence,
                          prototype_values: Sequence) -> Dict[int, float]:
    """
    For each prototype, fraction of its nearest samples whose value (expert
    type, or binary class) equals the prototype's value.
    """
    vals = np.asarray(sample_values)
    return {pid: float((vals[rows] == prototype_values[pid]).mean()) if len(rows) else float("nan")
            for pid, rows in neighbours.items()}


def per_prototype_confusion(assignment: Sequence[int], y_true: Sequence[int],
                            y_pred: Sequence[int], k_per_class: Sequence[int]) -> pd.DataFrame:
    """TP / FP / FN / TN counts grouped by nearest prototype."""
    df = pd.DataFrame({"proto": np.asarray(assignment), "y": np.asarray(y_true),
                       "p": np.asarray(y_pred)})
    pcls = prototype_classes(k_per_class)
    rows = []
    for pid in range(len(pcls)):
        g = df[df.proto == pid]
        rows.append({"proto": pid, "proto_class": int(pcls[pid]), "n": len(g),
                     "tp": int(((g.y == 1) & (g.p == 1)).sum()),
                     "fp": int(((g.y == 0) & (g.p == 1)).sum()),
                     "fn": int(((g.y == 1) & (g.p == 0)).sum()),
                     "tn": int(((g.y == 0) & (g.p == 0)).sum())})
    return pd.DataFrame(rows)
