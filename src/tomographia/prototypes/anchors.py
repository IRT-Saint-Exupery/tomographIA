"""
Embedding extraction and expert-anchor prototype initialisation.

An *anchor* is a training patch an expert assigned to a defect **type**
(e.g. negatives: air / matrix / matrix+air; positives: pores / lines /
pores+lines).  Each type yields one prototype, initialised at the anchor
closest to the mean of that type's anchors in the standardised embedding
space (a real sample, not an average; called the type's *medoid* below).
During training the anchors are pulled to their prototype (``L_anchor``)
and prototypes are kept near the medoids (``L_medoid``).

Anchor tables have one row per anchor with columns ``index`` (global sample
index), ``cls`` (class) and ``type`` (expert type id).  Anchors whose patch
label disagrees with their class are dropped, and anchors outside the
training split are ignored.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, loader, device: str = "cpu"
                       ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Embed every sample of *loader* with ``model.features``.

    Use a loader **without** augmentation and with ``shuffle=False``.

    Returns
    -------
    ``(Z [N, D], labels [N], indices [N] or None)``.
    """
    model.eval().to(device)
    feats, labels, idx = [], [], []
    for batch in loader:
        feats.append(model.features(batch[0].to(device)).cpu().numpy())
        labels.append(batch[1].numpy().ravel())
        if len(batch) > 2:
            idx.append(np.asarray(batch[2]).ravel())
    return (np.concatenate(feats), np.concatenate(labels).astype(int),
            np.concatenate(idx) if idx else None)


def fit_embedding_scaler(train_embeddings: np.ndarray) -> StandardScaler:
    """StandardScaler fitted on *training* embeddings only."""
    return StandardScaler().fit(train_embeddings)


@dataclass
class AnchorSet:
    """
    Prototype initialisation and per-sample anchor targets.

    Attributes
    ----------
    centroids_by_class : ``{class: [K_c, D]}`` medoid prototypes (standardised).
    type_order         : ``{class: [type ids]}`` — prototype ``k`` of class
                         ``c`` corresponds to ``type_order[c][k]``.
    medoid_rows        : ``{class: [row in Z_train]}`` of each medoid.
    anchor_class       : ``[n_total]`` long, class of each anchor sample, -1 otherwise.
    anchor_proto       : ``[n_total]`` long, local prototype id, -1 otherwise.
    """
    centroids_by_class: Dict[int, np.ndarray]
    type_order: Dict[int, List]
    medoid_rows: Dict[int, List[int]]
    anchor_class: torch.Tensor
    anchor_proto: torch.Tensor

    @property
    def k_per_class(self) -> List[int]:
        return [len(self.type_order[c]) for c in sorted(self.type_order)]

    @property
    def medoids(self) -> np.ndarray:
        """``[K, D]`` stacked medoids in prototype order (target of ``L_medoid``)."""
        return np.vstack([self.centroids_by_class[c] for c in sorted(self.centroids_by_class)])


def prototypes_from_anchors(
    z_train_scaled: np.ndarray,
    train_index: np.ndarray,
    anchors: pd.DataFrame,
    n_total: int,
    labels: Optional[pd.Series] = None,
    index_col: str = "index",
    class_col: str = "cls",
    type_col: str = "type",
) -> AnchorSet:
    """
    Build medoid prototypes and anchor targets from an expert anchor table.

    Parameters
    ----------
    z_train_scaled : ``[N_train, D]`` standardised training embeddings.
    train_index    : ``[N_train]`` global sample index of each row of
                     ``z_train_scaled`` (the patch DataFrame index).
    anchors        : one row per anchor with the global sample index, the
                     class (0/1) and the expert type id.
    n_total        : size of the global index space (``len(splits)``); the
                     anchor target tensors are indexed by global index.
    labels         : optional global ``label`` Series; anchors whose label
                     differs from their class are dropped.

    Returns
    -------
    :class:`AnchorSet`.
    """
    a = anchors[[index_col, class_col, type_col]].copy()
    a.columns = ["index", "cls", "type"]
    a["index"] = a["index"].astype(int)
    a["cls"] = a["cls"].astype(int)
    a = a.drop_duplicates()
    conflict = a["index"].duplicated(keep=False)
    if conflict.any():
        raise ValueError(f"Anchors listed with different classes or types: "
                         f"{sorted(a.loc[conflict, 'index'].unique())[:10]}")

    train_index = np.asarray(train_index).astype(int)
    if len(np.unique(train_index)) != len(train_index):
        raise ValueError("train_index contains duplicates: the patch DataFrame index must be "
                         "unique (use reset_index(drop=True) before splitting)")
    for name, ids in (("train_index", train_index), ("anchor index", a["index"].to_numpy())):
        if len(ids) and (ids.min() < 0 or ids.max() >= n_total):
            raise ValueError(f"{name} values must lie in [0, n_total={n_total}); got "
                             f"[{ids.min()}, {ids.max()}]. Pass n_total = splits.index.max() + 1 "
                             "or reset the DataFrame index before splitting")
    if labels is not None and not labels.index.is_unique:
        raise ValueError("labels index must be unique")

    if labels is not None:
        ok = labels.reindex(a["index"]).to_numpy() == a["cls"].to_numpy()
        if (~ok).any():
            logger.warning("Dropping %d anchors whose label disagrees with their class", int((~ok).sum()))
        a = a[ok]

    row_of = {int(g): i for i, g in enumerate(np.asarray(train_index))}
    a["row"] = a["index"].map(row_of)
    n_outside = int(a["row"].isna().sum())
    if n_outside:
        logger.info("%d anchors are outside the training split and are ignored", n_outside)
    a = a.dropna(subset=["row"])
    a["row"] = a["row"].astype(int)

    centroids, type_order, medoid_rows = {}, {}, {}
    anchor_class = torch.full((n_total,), -1, dtype=torch.long)
    anchor_proto = torch.full((n_total,), -1, dtype=torch.long)

    for c in sorted(a["cls"].unique()):
        ac = a[a["cls"] == c]
        types = sorted(ac["type"].unique())
        mats, rows = [], []
        for k, t in enumerate(types):
            r = ac.loc[ac["type"] == t, "row"].to_numpy()
            emb = z_train_scaled[r]
            best = int(r[np.argmin(np.linalg.norm(emb - emb.mean(0), axis=1))])
            mats.append(z_train_scaled[best])
            rows.append(best)
            g = torch.tensor(ac.loc[ac["type"] == t, "index"].to_numpy(), dtype=torch.long)
            anchor_class[g] = int(c)
            anchor_proto[g] = k
        centroids[int(c)] = np.stack(mats)
        type_order[int(c)] = types
        medoid_rows[int(c)] = rows

    found = sorted(centroids)
    if found != list(range(len(found))) or len(found) < 2:
        raise ValueError(f"Anchors (inside the training split) must cover every class 0..C-1 "
                         f"(C >= 2); found classes {found}")
    return AnchorSet(centroids, type_order, medoid_rows, anchor_class, anchor_proto)
