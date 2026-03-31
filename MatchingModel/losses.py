"""
losses.py — Loss functions for ToothMatchNet.

Available losses:
    ● BCELoss           — standard binary cross-entropy (with logits)
    ● FocalLoss         — addresses class imbalance by down-weighting easy negatives
    ● BCEFocalLoss      — weighted combination of BCE + Focal
    ● LabelSmoothingBCE — BCE with label smoothing to prevent over-confidence

All losses accept raw logits (before sigmoid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al. 2017).

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha:   Weighting factor for positive class (scalar in [0, 1]).
        gamma:   Focusing parameter (γ ≥ 0). γ=0 → standard BCE.
        reduction: "mean" | "sum" | "none"
        pos_weight: Optional per-sample positive class weight (like BCEWithLogitsLoss).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean",
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.reduction  = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B]  raw logits
            targets: [B]  binary labels {0.0, 1.0}
        """
        # Move pos_weight to the same device as logits (fixes CPU/GPU mismatch)
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        # Standard BCE (unreduced)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=pw
        )

        # p_t: probability of the true class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Label-Smoothed BCE
# ---------------------------------------------------------------------------

class LabelSmoothingBCE(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing.
    Soft targets: ỹ = (1 - ε) * y + ε/2
    """

    def __init__(self, smoothing: float = 0.05,
                 pos_weight: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing  = smoothing
        self.pos_weight = pos_weight
        self.reduction  = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        smooth_targets = (1.0 - self.smoothing) * targets + self.smoothing * 0.5
        # Move pos_weight to the same device as logits (fixes CPU/GPU mismatch)
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits, smooth_targets,
            pos_weight = pw,
            reduction  = self.reduction,
        )


# ---------------------------------------------------------------------------
# BCE + Focal combined loss
# ---------------------------------------------------------------------------

class BCEFocalLoss(nn.Module):
    """
    Weighted combination: loss = w_bce * BCE + w_focal * Focal

    Having both stabilises training (BCE provides gradients even when focal
    weight is near zero) while focal focuses on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 smoothing: float = 0.05,
                 w_bce: float = 0.5, w_focal: float = 0.5,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.w_bce   = w_bce
        self.w_focal = w_focal
        self.bce   = LabelSmoothingBCE(smoothing=smoothing,
                                        pos_weight=pos_weight)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma,
                                pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.w_bce * self.bce(logits, targets) + \
               self.w_focal * self.focal(logits, targets)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(cfg, pos_weight: Optional[float] = None) -> nn.Module:
    """
    Build the loss function from TrainConfig.

    Args:
        cfg:        Full Config or TrainConfig object.
        pos_weight: Override positive class weight (# neg / # pos).
                    Ignored if cfg.train.pos_weight is set.
    """
    tc = cfg.train if hasattr(cfg, "train") else cfg

    # Resolve pos_weight tensor
    pw_scalar = tc.pos_weight if tc.pos_weight is not None else pos_weight
    pw_tensor = torch.tensor([pw_scalar], dtype=torch.float32) \
                if pw_scalar is not None else None

    lt = tc.loss_type.lower()

    if lt == "bce":
        return LabelSmoothingBCE(
            smoothing  = tc.label_smoothing,
            pos_weight = pw_tensor,
        )
    elif lt == "focal":
        return FocalLoss(
            alpha      = tc.focal_alpha,
            gamma      = tc.focal_gamma,
            pos_weight = pw_tensor,
        )
    elif lt == "bce_focal":
        return BCEFocalLoss(
            alpha      = tc.focal_alpha,
            gamma      = tc.focal_gamma,
            smoothing  = tc.label_smoothing,
            pos_weight = pw_tensor,
        )
    else:
        raise ValueError(f"Unknown loss_type: {tc.loss_type}. "
                         f"Choose from: bce, focal, bce_focal")
