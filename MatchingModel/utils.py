"""
utils.py — Utility functions for ToothMatchNet.

Covers:
    ● Reproducibility (seeding)
    ● Checkpoint save / load
    ● Metric computation (accuracy, precision, recall, F1, AUC)
    ● Logging helpers
    ● LR scheduler factory
    ● Optimizer factory
    ● AverageMeter
    ● Top-k checkpoint management
    ● Model summary
"""

import os
import re
import math
import json
import random
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau, LambdaLR
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "ToothMatchNet",
               log_file: Optional[Path] = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:   # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    logger.propagate = False  # 禁用日志传播，避免重复输出
    
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks a running mean of any scalar metric."""

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray,
                    labels: np.ndarray,
                    probs: Optional[np.ndarray] = None,
                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        preds:     [N] binary predictions {0, 1}
        labels:    [N] binary ground-truth labels {0, 1}
        probs:     [N] predicted probabilities (for AUC/AP)
        threshold: decision threshold (informational only here)

    Returns:
        dict with keys: accuracy, precision, recall, f1, specificity,
                        auc (if probs provided), avg_precision (if probs provided)
    """
    eps = 1e-8
    preds  = preds.astype(int)
    labels = labels.astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    accuracy    = (tp + tn) / (tp + fp + tn + fn + eps)
    precision   = tp / (tp + fp + eps)
    recall      = tp / (tp + fn + eps)       # sensitivity
    specificity = tn / (tn + fp + eps)
    f1          = 2 * precision * recall / (precision + recall + eps)

    metrics = {
        "accuracy":    float(accuracy),
        "precision":   float(precision),
        "recall":      float(recall),
        "specificity": float(specificity),
        "f1":          float(f1),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }

    if probs is not None:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            probs_arr  = np.array(probs).ravel().astype(float)
            labels_int = (np.array(labels).ravel() >= 0.5).astype(int)

            # 必须有两类才能算 AUC
            unique_cls = np.unique(labels_int)
            if len(unique_cls) < 2:
                metrics["auc"]           = float("nan")
                metrics["avg_precision"] = float("nan")
            else:
                # 检查 probs 是否全相同（模型完全无区分度）
                if np.std(probs_arr) < 1e-8:
                    metrics["auc"]           = float("nan")
                    metrics["avg_precision"] = float("nan")
                else:
                    metrics["auc"]           = float(roc_auc_score(labels_int, probs_arr))
                    metrics["avg_precision"] = float(average_precision_score(labels_int, probs_arr))
        except ImportError:
            pass   # sklearn optional
        except Exception as e:
            import traceback
            print(f"[utils] AUC computation failed: {e}")
            traceback.print_exc()
            metrics["auc"]           = float("nan")
            metrics["avg_precision"] = float("nan")
            # 打印具体原因，方便排查
            import warnings
            warnings.warn(f"[compute_metrics] AUC computation failed: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg) -> Optimizer:
    """
    Build optimizer with differential learning rates.

    Parameter groups:
        "backbone"    — tooth_encoder.backbone + eden_encoder.backbone
                        lr = learning_rate * backbone_lr_scale (small, fine-tuning)
        "head+fusion" — adapters + fusion + head
                        lr = learning_rate (full)

    The adapter (4ch→3ch) is always in the high-LR group because it is
    randomly initialized and needs to learn fast regardless of freeze state.
    """
    tc = cfg.train if hasattr(cfg, "train") else cfg

    # Collect backbone params from both branches (if model has dual encoders)
    backbone_params = []
    if hasattr(model, "tooth_encoder") and hasattr(model, "eden_encoder"):
        # New dual-branch architecture
        backbone_params = (
            list(model.tooth_encoder.backbone.parameters()) +
            list(model.eden_encoder.backbone.parameters())
        )
    elif hasattr(model, "encoder"):
        # Legacy single-encoder architecture (fallback)
        backbone_params = list(model.encoder.parameters())

    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters()
                    if id(p) not in backbone_param_ids]

    param_groups = [
        {"params": backbone_params, "lr": tc.learning_rate * tc.backbone_lr_scale,
         "name": "backbone"},
        {"params": other_params,    "lr": tc.learning_rate,
         "name": "head+fusion"},
    ]

    opt_name = tc.optimizer.lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=tc.weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=tc.weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9,
                                weight_decay=tc.weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {tc.optimizer}")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: Optimizer, cfg,
                    steps_per_epoch: int = 1):
    """
    Build LR scheduler. Warmup is implemented via a LambdaLR pre-scheduler.

    Returns:
        scheduler  — the main scheduler object
        warmup_scheduler — LambdaLR for warmup (or None)
    """
    tc = cfg.train if hasattr(cfg, "train") else cfg

    # Warmup scheduler
    warmup_scheduler = None
    if tc.warmup_epochs > 0:
        def warmup_lambda(current_epoch):
            if current_epoch < tc.warmup_epochs:
                return float(current_epoch + 1) / float(tc.warmup_epochs + 1)
            return 1.0
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    sched_name = tc.scheduler.lower()
    if sched_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max   = tc.epochs - tc.warmup_epochs,
            eta_min = tc.min_lr,
        )
    elif sched_name == "step":
        scheduler = StepLR(optimizer,
                            step_size = tc.step_size,
                            gamma     = tc.gamma)
    elif sched_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max",
            factor   = tc.gamma,
            patience = 5,
            min_lr   = tc.min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler: {tc.scheduler}")

    return scheduler, warmup_scheduler


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Saves model checkpoints and keeps only the top-k best ones
    (ranked by a monitored metric, higher = better by default).

    Checkpoint filename format:
        epoch_{epoch:04d}_val_{metric_name}_{value:.4f}.pth
    """

    def __init__(self, checkpoint_dir: Path,
                 top_k: int = 3,
                 metric_name: str = "f1",
                 higher_is_better: bool = True):
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.top_k           = top_k
        self.metric_name     = metric_name
        self.higher_is_better = higher_is_better
        self.best_checkpoints: List[Tuple[float, Path]] = []  # [(metric_val, path)]

    # ------------------------------------------------------------------
    def save(self, state: Dict[str, Any], metric_val: float,
             epoch: int, is_best: bool = False) -> Path:
        """
        Save checkpoint. Prune old ones if we exceed top_k.
        Returns the saved checkpoint path.
        """
        fname = (f"epoch_{epoch:04d}"
                 f"_val_{self.metric_name}_{metric_val:.4f}.pth")
        ckpt_path = self.checkpoint_dir / fname
        torch.save(state, ckpt_path)

        # Track checkpoints
        self.best_checkpoints.append((metric_val, ckpt_path))
        # Sort: best first
        self.best_checkpoints.sort(
            key=lambda x: x[0], reverse=self.higher_is_better
        )

        # Remove excess checkpoints
        while len(self.best_checkpoints) > self.top_k:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

        # Symlink "best.pth" to the current best checkpoint
        if is_best or self.best_checkpoints[0][1] == ckpt_path:
            best_link = self.checkpoint_dir / "best.pth"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(ckpt_path.name)

        return ckpt_path

    # ------------------------------------------------------------------
    def load_best(self, device: str = "cuda") -> Optional[Dict[str, Any]]:
        """Load the best checkpoint (via best.pth symlink)."""
        best_link = self.checkpoint_dir / "best.pth"
        if best_link.exists():
            return torch.load(best_link, map_location=device)
        # Fallback: most recently modified .pth
        pths = sorted(self.checkpoint_dir.glob("*.pth"),
                      key=lambda p: p.stat().st_mtime)
        if pths:
            return torch.load(pths[-1], map_location=device)
        return None

    # ------------------------------------------------------------------
    @property
    def best_metric(self) -> Optional[float]:
        if self.best_checkpoints:
            return self.best_checkpoints[0][0]
        return None


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path,
                    model: nn.Module,
                    optimizer: Optional[Optimizer] = None,
                    device: str = "cuda") -> Dict[str, Any]:
    """Load checkpoint and restore model (+ optionally optimizer) state."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training when a monitored metric stops improving."""

    def __init__(self, patience: int = 20,
                 higher_is_better: bool = True,
                 min_delta: float = 1e-5):
        self.patience         = patience
        self.higher_is_better = higher_is_better
        self.min_delta        = min_delta
        self.best_value       = None
        self.counter          = 0
        self.should_stop      = False

    def step(self, value: float) -> bool:
        """
        Call once per epoch.
        Returns True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.higher_is_better:
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter    = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False

    def reset(self) -> None:
        self.best_value  = None
        self.counter     = 0
        self.should_stop = False


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def model_summary(model: nn.Module,
                  input_shapes: Optional[List[Tuple]] = None) -> str:
    """Print a brief model summary."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        "=" * 55,
        f"  Model: {model.__class__.__name__}",
        "-" * 55,
        f"  Total parameters    : {total:>15,}",
        f"  Trainable parameters: {trainable:>15,}",
        f"  Frozen  parameters  : {total - trainable:>15,}",
    ]
    if input_shapes:
        lines.append(f"  Input shapes        : {input_shapes}")
    lines.append("=" * 55)
    summary = "\n".join(lines)
    return summary


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def count_samples_per_class(dataset) -> Dict[str, int]:
    labels = dataset.labels
    return {
        "total":    len(labels),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
    }


def tensor_to_device(batch: Dict[str, Any],
                     device: torch.device) -> Dict[str, Any]:
    """Move all tensor values in a batch dict to device."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
