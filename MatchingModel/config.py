"""
config.py — Centralized configuration for ToothMatchNet.
All model, training, and inference hyper-parameters live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR        = Path(__file__).resolve().parent.parent          # ToothMatchNet/
DATA_DIR        = ROOT_DIR / "MatchingData"
TRAIN_DIR       = DATA_DIR / "train"
VAL_DIR         = DATA_DIR / "val"
TEST_DIR        = DATA_DIR / "test"
LABELS_CSV      = DATA_DIR / "labels.csv"
CHECKPOINT_DIR  = ROOT_DIR / "MatchingCheckpoints"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # ConvNeXt variant: "convnext_small" | "convnext_base" | "convnext_tiny"
    backbone: str           = "convnext_small"

    # Number of input channels per branch (depth 1ch + normal 3ch = 4ch)
    # NOTE: model uses InputAdapter(4→3) so backbone always sees 3ch
    in_channels: int        = 4

    # Whether to load ImageNet-pretrained weights for the backbone.
    # InputAdapter(4→3) is always randomly initialized (fresh learnable layer).
    # All ConvNeXt layers keep their pretrained weights unchanged.
    pretrained: bool        = True

    # Cross-attention
    attn_embed_dim: int     = 256   # projected feature dimension
    attn_num_heads: int     = 8
    attn_dropout: float     = 0.1
    attn_num_layers: int    = 2     # stacked cross-attention blocks

    # Classification head
    head_hidden_dims: Tuple[int, ...] = (256, 64)
    head_dropout: float     = 0.3

    # Output: 1 logit → sigmoid → probability of "match"
    num_classes: int        = 1

    # Freeze backbone for N epochs → train adapter + head + fusion first
    # then unfreeze backbone for fine-tuning.
    # Recommended: 10 epochs freeze → backbone learns from stable head gradient
    freeze_backbone_epochs: int = 10


# ---------------------------------------------------------------------------
# Augmentation / Data configuration
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    image_size: Tuple[int, int]     = (224, 224)

    # --- Tooth branch augmentations (direction is FIXED, U-shape) ---
    tooth_hflip_prob: float         = 0.5    # horizontal flip (symmetric arch)
    tooth_vflip_prob: float         = 0.0    # no vertical flip (U always opens down)
    tooth_rotate_degrees: float     = 10.0   # small rotation only (arch direction fixed)
    # Scale jitter: simulate different tooth arch sizes in the canvas.
    # 0.15 → randomly zoom [0.85×, 1.15×] so model sees arches of varying fill ratio.
    tooth_scale_jitter: float       = 0.15
    tooth_color_jitter: bool        = True
    tooth_color_jitter_brightness: float = 0.2
    tooth_color_jitter_contrast: float   = 0.2

    # --- Eden branch augmentations (orientation is ARBITRARY → full rotation) ---
    eden_rotate_degrees: float      = 180.0  # full ±180° rotation (arbitrary direction)
    eden_hflip_prob: float          = 0.5    # flip also covers reflections
    eden_vflip_prob: float          = 0.5
    # Scale jitter for eden: same range so model sees consistent size variance
    eden_scale_jitter: float        = 0.15
    eden_color_jitter: bool         = True
    eden_color_jitter_brightness: float  = 0.2
    eden_color_jitter_contrast: float    = 0.2

    # Common
    normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)
    normalize_std:  Tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)

    # DataLoader
    num_workers: int                = 4
    pin_memory: bool                = True
    prefetch_factor: int            = 2


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Basic
    seed: int                       = 42
    epochs: int                     = 150
    batch_size: int                 = 16

    # Optimizer — AdamW with differential LR
    optimizer: str                  = "adamw"
    learning_rate: float            = 3e-4    # head + fusion + adapter LR
    weight_decay: float             = 1e-2
    # backbone LR = learning_rate * backbone_lr_scale (after unfreeze)
    backbone_lr_scale: float        = 0.1     # backbone gets 3e-5

    # Scheduler
    scheduler: str                  = "cosine"
    warmup_epochs: int              = 5
    min_lr: float                   = 1e-6
    # StepLR params (only used when scheduler == "step")
    step_size: int                  = 30
    gamma: float                    = 0.1

    # Loss
    loss_type: str                  = "bce_focal"
    focal_alpha: float              = 0.25
    focal_gamma: float              = 2.0
    label_smoothing: float          = 0.05

    # Class imbalance handling
    use_weighted_sampler: bool      = True
    pos_weight: Optional[float]     = None     # None → auto from dataset

    # Regularisation
    mixup_alpha: float              = 0.0      # disabled for small dataset
    cutmix_alpha: float             = 0.0

    # Gradient
    grad_clip_norm: float           = 1.0
    accumulation_steps: int         = 1        # set >1 only if GPU OOM

    # AMP (automatic mixed precision)
    use_amp: bool                   = True

    # Checkpointing
    save_every_n_epochs: int        = 5
    keep_top_k_checkpoints: int     = 3
    early_stopping_patience: int    = 30

    # Logging
    log_every_n_steps: int          = 10
    use_tensorboard: bool           = True

    # Device
    device: str                     = "cuda"
    multi_gpu: bool                 = False


# ---------------------------------------------------------------------------
# Inference / Prediction configuration
# ---------------------------------------------------------------------------
@dataclass
class InferConfig:
    checkpoint_path: Optional[str]  = None      # if None → auto-load best ckpt
    threshold: float                = 0.5       # sigmoid output threshold
    batch_size: int                 = 16
    device: str                     = "cuda"
    use_amp: bool                   = True

    # Test-time augmentation for eden branch (average over N rotations)
    tta_enabled: bool               = True
    tta_rotations: Tuple[float, ...] = (0, 90, 180, 270)
    tta_hflip: bool                 = True


# ---------------------------------------------------------------------------
# Convenience: grouped config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model:  ModelConfig  = field(default_factory=ModelConfig)
    data:   DataConfig   = field(default_factory=DataConfig)
    train:  TrainConfig  = field(default_factory=TrainConfig)
    infer:  InferConfig  = field(default_factory=InferConfig)


# Singleton default instance (importable directly)
CFG = Config()
