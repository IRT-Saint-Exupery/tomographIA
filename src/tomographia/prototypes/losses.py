"""
Composite prototype-network loss.

Terms (all on the normalised squared distances ``dist2`` of the head)
---------------------------------------------------------------------
``class``     cross-entropy on class logits (optional class weights)
``pull``      usage-aware soft-min distance of each sample to the prototypes
              of its class (rarely used prototypes are up-weighted)
``push``      ``exp(-d / tau_push)`` repulsion from other-class prototypes
``div``       within-class prototype diversity: hinge ``relu(δ - ||p̂_i - p̂_j||)²``
              on L2-normalised prototypes
``within_ent`` KL(within-class assignment ‖ uniform), favours soft assignments
``usage``     MSE between average within-class usage and uniform usage
``anchor``    NLL of each expert anchor's designated prototype
``proto_l2``  L2 on prototype coordinates;  ``tau_reg``: ``|τ - 1|``
``medoid``    MSE between prototypes and their (fixed) anchor medoids

All weights live in :class:`ProtoLossWeights`; ``prototype_loss`` also
returns the medoid term when ``medoids`` is given, so training and
validation share one objective.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .model import PrototypeHead


@dataclass
class ProtoLossWeights:
    """Loss weights."""
    w_class: float = 0.1
    w_pull: float = 0.05
    w_push: float = 0.01
    w_div: float = 1.0
    w_within_entropy: float = 0.01
    w_usage: float = 0.1
    w_anchor: float = 2.0
    w_medoid: float = 0.5
    w_proto_l2: float = 1e-6
    w_tau_reg: float = 1e-4

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def medoid_regulariser(head: PrototypeHead, medoids: torch.Tensor) -> torch.Tensor:
    """MSE between prototypes and their reference medoids ``[K, D]``."""
    return F.mse_loss(head.prototypes, medoids.to(head.prototypes.device))


def prototype_loss(
    out: Dict[str, torch.Tensor],
    y: torch.Tensor,
    head: PrototypeHead,
    weights: Optional[ProtoLossWeights] = None,
    class_weights: Optional[torch.Tensor] = None,
    push_type: str = "exp",
    tau_push: float = 0.1,
    margin_push: float = 0.5,
    delta: float = 0.7,
    usage_gamma: float = 0.5,
    sample_indices: Optional[torch.Tensor] = None,
    anchor_class: Optional[torch.Tensor] = None,
    anchor_proto: Optional[torch.Tensor] = None,
    medoids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute all loss terms and their weighted ``total``.

    Parameters
    ----------
    out            : output dict of :class:`PrototypeNetwork` / :class:`PrototypeHead`.
    y              : ``[B]`` long class labels.
    class_weights  : ``[C]`` weights for the cross-entropy term.
    push_type      : ``"exp"`` (default) or ``"hinge"`` (margin ``margin_push``).
    delta          : diversity margin on normalised prototypes.
    usage_gamma    : strength of inverse-usage re-weighting in the pull term.
    sample_indices, anchor_class, anchor_proto : global indices of the batch
                     and anchor targets (see :class:`AnchorSet`); the anchor
                     term is 0 when omitted (e.g. on validation).
    medoids        : ``[K, D]`` reference medoids for the medoid term.

    Returns
    -------
    dict of detached terms plus the differentiable ``"total"``.
    """
    w = weights or ProtoLossWeights()
    device = out["class_logits"].device
    class_logits, proto_probs, dist2 = out["class_logits"], out["proto_probs"], out["dist2"]
    C, K = class_logits.shape[1], proto_probs.shape[1]
    slices = head.class_slices
    eps = 1e-8
    zero = torch.zeros((), device=device)

    # 1) classification
    cw = class_weights.to(device) if class_weights is not None else None
    L_class = F.cross_entropy(class_logits, y, weight=cw)

    # 2) usage-aware pull, within-class entropy, usage
    L_pull, L_within, L_usage = zero, zero, zero
    for c in range(C):
        m = y == c
        if not m.any():
            continue
        p_w = proto_probs[m, slices[c]]
        p_w = p_w / (p_w.sum(1, keepdim=True) + eps)
        usage = p_w.mean(0)
        usage = usage / (usage.sum() + eps)
        inv = (usage + 1e-3).pow(-usage_gamma)
        inv = (inv / inv.sum()).detach()
        L_pull = L_pull + (p_w * inv[None, :] * dist2[m, slices[c]]).sum(1).mean()
        k_c = p_w.shape[1]
        pc = p_w.clamp_min(eps)
        L_within = L_within + (pc * (pc.log() + torch.log(torch.tensor(float(k_c))))).sum(1).mean()
        L_usage = L_usage + F.mse_loss(usage, torch.full_like(usage, 1.0 / k_c))

    # 3) expert anchors
    L_anchor = zero
    if sample_indices is not None and anchor_class is not None and anchor_proto is not None:
        sample_indices = sample_indices.to(anchor_class.device)
        a_cls = anchor_class[sample_indices].to(device)
        a_proto = anchor_proto[sample_indices].to(device)
        for c in range(C):
            m = (y == c) & (a_cls == c)
            if m.any():
                p_w = proto_probs[m, slices[c]]
                L_anchor = L_anchor + F.nll_loss((p_w + eps).log(), a_proto[m])

    # 4) push away from other-class prototypes
    L_push = zero
    for c in range(C):
        m = y == c
        if not m.any():
            continue
        for c_opp in range(C):
            if c_opp == c:
                continue
            d = dist2[m, slices[c_opp]]
            if push_type == "exp":
                L_push = L_push + torch.exp(-d / tau_push).mean()
            elif push_type == "hinge":
                L_push = L_push + F.relu(margin_push - d).mean()
            else:
                raise ValueError("push_type must be 'exp' or 'hinge'")

    # 5) within-class diversity on normalised prototypes
    P = head.prototypes
    PD = torch.cdist(F.normalize(P, dim=1), F.normalize(P, dim=1))
    L_div = zero
    for s in slices:
        k = s.stop - s.start
        if k < 2:
            continue
        block = PD[s, s]
        off = ~torch.eye(k, dtype=torch.bool, device=P.device)
        L_div = L_div + F.relu(delta - block[off]).pow(2).mean()

    # 6) small regularisers
    L_proto_l2 = P.pow(2).mean()
    L_tau = (torch.clamp(head.tau, min=1e-6) - 1.0).abs()
    L_medoid = medoid_regulariser(head, medoids) if medoids is not None else zero

    total = (w.w_class * L_class + w.w_pull * L_pull + w.w_push * L_push
             + w.w_div * L_div + w.w_within_entropy * L_within + w.w_usage * L_usage
             + w.w_anchor * L_anchor + w.w_proto_l2 * L_proto_l2
             + w.w_tau_reg * L_tau + w.w_medoid * L_medoid)

    terms = {"class": L_class, "pull": L_pull, "push": L_push, "div": L_div,
             "within_ent": L_within, "usage": L_usage, "anchor": L_anchor,
             "proto_l2": L_proto_l2, "tau_reg": L_tau, "medoid": L_medoid}
    result = {k: v.detach() for k, v in terms.items()}
    result["total"] = total
    return result
