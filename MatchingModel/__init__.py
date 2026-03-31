"""
ToothMatchNet.MatchingModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core package for the ToothMatchNet dental matching pipeline.
"""

from .config  import CFG, Config, ModelConfig, DataConfig, TrainConfig, InferConfig
from .model   import ToothMatchNet, build_model
from .dataset import ToothMatchDataset, build_dataloaders, BranchAugmentor, Normalize4ch
from .losses  import FocalLoss, LabelSmoothingBCE, BCEFocalLoss, build_loss
from .utils   import (
    set_seed, get_logger, AverageMeter, compute_metrics,
    build_optimizer, build_scheduler,
    CheckpointManager, EarlyStopping,
    model_summary,
)

__all__ = [
    # Config
    "CFG", "Config", "ModelConfig", "DataConfig", "TrainConfig", "InferConfig",
    # Model
    "ToothMatchNet", "build_model",
    # Dataset
    "ToothMatchDataset", "build_dataloaders", "BranchAugmentor", "Normalize4ch",
    # Losses
    "FocalLoss", "LabelSmoothingBCE", "BCEFocalLoss", "build_loss",
    # Utils
    "set_seed", "get_logger", "AverageMeter", "compute_metrics",
    "build_optimizer", "build_scheduler",
    "CheckpointManager", "EarlyStopping",
    "model_summary",
]
