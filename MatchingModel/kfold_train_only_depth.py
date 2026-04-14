"""
kfold_train_only_depth.py — K-Fold Cross Validation Training (Depth-Only)

v3 Fix log (相对 v2):
    [CRITICAL] 验证集从不使用 augmentation（v2 的 val_ds 和 train_ds 共用同一个带
               augmentor 的 dataset，导致 val 结果随机且评估失真）
    [CRITICAL] 降低 head_dropout: 0.3 → 0.1（高 dropout 使模型在 train 模式依靠随机
               噪声产生方差，eval 模式 dropout 关闭后 → 输出常数，即 val σ=0）
    [CRITICAL] focal_alpha: 0.25 → 0.5（数据集接近平衡 47%/53%，alpha=0.25 把正类权重
               压到 0.25 与 pos_weight=1.1 方向冲突，应用 0.5 保持中立）
    [FIX]      scale_feature 从 1 维扩展到 3 维：
               [log(tooth_scale), log(eden_scale), log(tooth/eden)]
               同时保留绝对大小信息，让模型判断"两张图各自有多大"
    [FIX]      不冻结 backbone（depth 图和 ImageNet RGB 分布差异极大，冻结 backbone 等
               于强迫模型用 ImageNet 的"随机特征"来学匹配）
    [FIX]      attn_dropout + head_dropout 通过 CLI 暴露可调
    [FIX]      build_depth_only_model 里重建 ClassificationHead 使 scale_dim=3 生效

v2 Fix log (相对原版，保留不变):
    batch_size: 1 → 8  /  BN1d → GroupNorm  /  真正执行冻结/解冻
    scale_feature 传给 model.forward()  /  pos_weight ≥ 1
    weight_decay: 5e-2 → 1e-2  /  drop_last=True  /  梯度累积
"""

import os
import sys
import argparse
import json
import random
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from MatchingModel.config import CFG, ModelConfig, DataConfig, TrainConfig, Config
from MatchingModel.model import (ToothMatchNet, InputAdapter, BranchEncoder,
                                  ClassificationHead, _BACKBONE_REGISTRY)
from MatchingModel.losses import build_loss
from MatchingModel.utils import (build_optimizer, build_scheduler, get_logger,
                                  save_checkpoint)


# ---------------------------------------------------------------------------
# Depth-only image loading
# ---------------------------------------------------------------------------

def load_depth_only(depth_path: Path,
                    image_size: Tuple[int, int],
                    keep_aspect: bool = True) -> Tuple[torch.Tensor, float]:
    target_h, target_w = image_size
    depth_pil = Image.open(depth_path).convert("L")
    orig_w, orig_h = depth_pil.size

    if keep_aspect:
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        depth_pil = depth_pil.resize((new_w, new_h), Image.BILINEAR)
        pad_top  = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        depth_t  = TF.to_tensor(depth_pil)
        depth_t  = F.pad(depth_t, [pad_left, target_w - new_w - pad_left,
                                    pad_top,  target_h - new_h - pad_top])
    else:
        scale    = min(target_w / orig_w, target_h / orig_h)
        depth_pil = depth_pil.resize((target_w, target_h), Image.BILINEAR)
        depth_t  = TF.to_tensor(depth_pil)

    return depth_t, scale


# ---------------------------------------------------------------------------
# 1-channel normalizer
# ---------------------------------------------------------------------------

class Normalize1ch:
    def __init__(self, mean=(0.5,), std=(0.5,)):
        self.mean = torch.tensor(mean).view(1, 1, 1)
        self.std  = torch.tensor(std).view(1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# Depth augmentor (1-channel)
# ---------------------------------------------------------------------------

class DepthAugmentor:
    def __init__(self, rotate_degrees=0.0, hflip_prob=0.0, vflip_prob=0.0,
                 scale_jitter=0.0, color_jitter=False,
                 brightness=0.0, contrast=0.0):
        self.rotate_degrees = rotate_degrees
        self.hflip_prob     = hflip_prob
        self.vflip_prob     = vflip_prob
        self.scale_jitter   = scale_jitter
        self.color_jitter   = color_jitter
        self.brightness     = brightness
        self.contrast       = contrast

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        H, W = img.shape[-2], img.shape[-1]

        if self.scale_jitter > 0:
            s     = random.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter)
            new_h = max(1, int(round(H * s)))
            new_w = max(1, int(round(W * s)))
            img   = F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                                  mode="bilinear", align_corners=False).squeeze(0)
            if s > 1.0:
                img = img[:, (new_h - H) // 2:(new_h - H) // 2 + H,
                              (new_w - W) // 2:(new_w - W) // 2 + W]
            else:
                pt, pl = (H - new_h) // 2, (W - new_w) // 2
                img = F.pad(img, [pl, W - new_w - pl, pt, H - new_h - pt])

        if self.rotate_degrees > 0:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            img   = TF.rotate(img, angle,
                              interpolation=TF.InterpolationMode.BILINEAR, fill=0)

        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            img = TF.hflip(img)
        if self.vflip_prob > 0 and random.random() < self.vflip_prob:
            img = TF.vflip(img)

        if self.color_jitter:
            if self.brightness > 0:
                b = random.uniform(1 - self.brightness, 1 + self.brightness)
                img = torch.clamp(img * b, 0, 1)
            if self.contrast > 0:
                c = random.uniform(1 - self.contrast, 1 + self.contrast)
                img = torch.clamp((img - 0.5) * c + 0.5, 0, 1)

        return img


# ---------------------------------------------------------------------------
# Depth-only dataset
# ---------------------------------------------------------------------------

SCALE_DIM = 3  # [log(tooth_scale), log(eden_scale), log(tooth/eden)]


class DepthOnlyDataset(Dataset):
    """
    每个样本返回的 scale_feature 是 3 维：
        [0] log(tooth_scale)         — tooth 图被缩放的绝对程度
        [1] log(eden_scale)          — eden 图被缩放的绝对程度
        [2] log(tooth_scale/eden)    — 两者的比值（原来唯一的维度）

    这样模型既知道"谁大谁小"，又知道"差了多少"，
    比单独传比值更能学到物理尺寸的匹配关系。
    """

    SUFFIXES = {
        "eden_depth":  "_eden_depth.png",
        "tooth_depth": "_tooth_depth.png",
    }

    def __init__(self, data_dir, labels_csv, image_size=(224, 224),
                 tooth_augmentor=None, eden_augmentor=None,
                 normalizer=None, split="train"):
        self.data_dir        = Path(data_dir)
        self.image_size      = image_size
        self.tooth_augmentor = tooth_augmentor
        self.eden_augmentor  = eden_augmentor
        self.normalizer      = normalizer
        self.split           = split
        self.samples         = self._load_samples(labels_csv)

    @staticmethod
    def _find_file(sample_dir, suffix):
        matches = sorted(p for p in sample_dir.iterdir()
                         if p.is_file() and p.name.endswith(suffix))
        if not matches:
            return None
        if len(matches) > 1:
            print(f"[Dataset] WARNING: multiple matches for '{suffix}' in "
                  f"{sample_dir}, using '{matches[0].name}'")
        return matches[0]

    def _load_samples(self, labels_csv):
        samples = []
        has_split_col = None
        with open(labels_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if all(v is None or str(v).strip() == "" for v in row.values()):
                    continue
                row = {k.strip().lstrip("\ufeff"): v for k, v in row.items()}
                if "sample_id" not in row or "label" not in row:
                    raise KeyError(f"CSV must have 'sample_id' and 'label'. "
                                   f"Got: {list(row.keys())}")
                if has_split_col is None:
                    has_split_col = "split" in row
                if has_split_col and row.get("split", "").strip() != self.split:
                    continue

                sid   = row["sample_id"].strip()
                label = int(row["label"].strip())
                sdir  = self.data_dir / sid
                if not sdir.is_dir():
                    print(f"[Dataset] WARNING: directory not found: {sdir}")
                    continue

                paths, missing = {}, []
                for key, sfx in self.SUFFIXES.items():
                    found = self._find_file(sdir, sfx)
                    if found is None:
                        missing.append(sfx)
                    else:
                        paths[key] = found
                if missing:
                    print(f"[Dataset] WARNING: missing {missing} for '{sid}' — skipped")
                    continue

                samples.append({"sample_id": sid, "label": label, **paths})

        if not samples:
            raise RuntimeError(
                f"No valid samples found for split='{self.split}' in {self.data_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        tooth_img, tooth_scale = load_depth_only(s["tooth_depth"], self.image_size)
        eden_img,  eden_scale  = load_depth_only(s["eden_depth"],  self.image_size)

        if self.tooth_augmentor:
            tooth_img = self.tooth_augmentor(tooth_img)
        if self.eden_augmentor:
            eden_img  = self.eden_augmentor(eden_img)
        if self.normalizer:
            tooth_img = self.normalizer(tooth_img)
            eden_img  = self.normalizer(eden_img)

        # FIX: 3 维 scale_feature 而不是 1 维
        # log(tooth_scale) / log(eden_scale) 保留各自绝对大小
        # log(ratio) 保留相对大小，与原来一致
        eps = 1e-6
        scale_feature = torch.tensor([
            float(np.log(max(tooth_scale, eps))),
            float(np.log(max(eden_scale, eps))),
            float(np.log(tooth_scale / max(eden_scale, eps))),
        ], dtype=torch.float32)

        return {
            "tooth_img":     tooth_img,
            "eden_img":      eden_img,
            "scale_feature": scale_feature,
            "label":         torch.tensor(s["label"], dtype=torch.float32),
            "sample_id":     s["sample_id"],
        }

    @property
    def labels(self):
        return [s["label"] for s in self.samples]


# ---------------------------------------------------------------------------
# GroupNorm replacement for BN1d
# ---------------------------------------------------------------------------

def replace_bn1d_with_gn(model: nn.Module, num_groups: int = 32) -> nn.Module:
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.BatchNorm1d):
            continue
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        nf   = module.num_features
        ng   = num_groups
        while ng > 1 and nf % ng != 0:
            ng -= 1
        gn = nn.GroupNorm(ng, nf, eps=module.eps, affine=module.affine)
        if module.affine:
            gn.weight.data.copy_(module.weight.data)
            gn.bias.data.copy_(module.bias.data)
        setattr(parent, attr, gn)
        print(f"  [GN] {name}: BN1d({nf}) → GroupNorm({ng},{nf})")
    return model


# ---------------------------------------------------------------------------
# Build depth-only model  (1ch input, scale_dim=3, head_dropout 可调)
# ---------------------------------------------------------------------------

def build_depth_only_model(cfg, head_dropout: float = 0.1) -> ToothMatchNet:
    """
    head_dropout 默认从 0.3 降到 0.1：
    高 dropout 使模型在 train 模式靠随机噪声产生方差，
    eval 模式 dropout 关闭后输出退化为常数（val σ=0 的根因）。
    """
    m = cfg.model if hasattr(cfg, "model") else cfg

    model = ToothMatchNet(
        backbone        = m.backbone,
        pretrained      = m.pretrained,
        attn_embed_dim  = m.attn_embed_dim,
        attn_num_heads  = m.attn_num_heads,
        attn_dropout    = m.attn_dropout,
        attn_num_layers = m.attn_num_layers,
        head_hidden_dims= m.head_hidden_dims,
        # 先用原始 head_dropout 构建，下面再替换 head
        head_dropout    = head_dropout,
    )

    # FIX: 1ch adapter
    model.tooth_encoder.adapter = InputAdapter(in_channels=1, out_channels=3)
    model.eden_encoder.adapter  = InputAdapter(in_channels=1, out_channels=3)

    # FIX: 重建 head，使 scale_dim=3 生效
    # 原始 head 是 scale_dim=1，现在需要 scale_dim=3
    model.head = ClassificationHead(
        in_dim      = m.attn_embed_dim * 2,
        hidden_dims = m.head_hidden_dims,
        dropout     = head_dropout,
        scale_dim   = SCALE_DIM,   # 3 维 scale feature
    )
    # 重新初始化新 head
    for layer in model.head.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # FIX: BN1d → GroupNorm（batch_size 敏感问题）
    print("[build_model] Replacing BatchNorm1d → GroupNorm:")
    model = replace_bn1d_with_gn(model, num_groups=32)

    return model


# ---------------------------------------------------------------------------
# Backbone freeze / unfreeze
# ---------------------------------------------------------------------------

def freeze_backbone(model: nn.Module) -> Tuple[int, int]:
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        is_non_bb = any(k in name for k in
                        ("adapter", "head", "fusion", "attn",
                         "proj_tooth", "proj_eden", "norm_tooth", "norm_eden"))
        if is_non_bb:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen, trainable


def unfreeze_all(model: nn.Module) -> int:
    for p in model.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters())


def reset_backbone_lr(optimizer, new_lr: float):
    for group in optimizer.param_groups:
        if group.get("name") == "backbone":
            group["lr"] = new_lr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="K-Fold Depth-Only Training (v3)")

    p.add_argument("--k-folds",       type=int,   default=5)
    p.add_argument("--fold",          type=int,   default=None)
    p.add_argument("--backbone",      type=str,   default="resnet50",
                   choices=list(_BACKBONE_REGISTRY.keys()))
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--accum-steps",   type=int,   default=1)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-2)

    # FIX: 默认不冻结 backbone（depth 图与 ImageNet 分布差异太大，
    #       冻结等于强迫模型用"ImageNet 的随机特征"来学匹配）
    p.add_argument("--freeze-epochs", type=int,   default=0,
                   help="冻结 backbone 的 epoch 数（depth-only 建议设为 0 不冻结）")

    # FIX: head_dropout 0.3 → 0.1
    p.add_argument("--head-dropout",  type=float, default=0.1,
                   help="分类头 dropout（原 0.3 过高，eval 关闭 dropout 后输出退化为常数）")
    # FIX: focal_alpha 0.25 → 0.5 for balanced dataset
    p.add_argument("--focal-alpha",   type=float, default=0.5,
                   help="Focal loss alpha（数据集接近平衡时用 0.5，原来的 0.25 严重压低正类权重）")

    p.add_argument("--data-dir",      type=str,   default=None)
    p.add_argument("--use-all-splits",action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output-dir",    type=str,
                   default="MatchingCheckpoints/KFold_DepthOnly")
    p.add_argument("--device",        type=str,   default="cuda")

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Dataset creation  — 返回 (train_dataset_with_aug, val_dataset_no_aug)
# ---------------------------------------------------------------------------

def create_datasets(cfg, matching_data_dir: Path, use_all_splits: bool = False):
    """
    FIX: 返回两个独立的 dataset 对象：
        train_ds: 带数据增强，用于训练子集
        val_ds:   无数据增强，用于验证子集

    原来 train/val 共用同一个带 augmentor 的 dataset（Subset 不拷贝），
    导致：
      1. val 结果随机（每次评估不同），评估指标不可信
      2. augmentation 的随机性干扰 val 特征，使 val σ 非零但无意义
    """
    dc = cfg.data
    normalizer = Normalize1ch(mean=(0.5,), std=(0.5,))
    labels_csv = matching_data_dir / "labels.csv"

    tooth_aug = DepthAugmentor(
        rotate_degrees=dc.tooth_rotate_degrees,
        hflip_prob=dc.tooth_hflip_prob,
        vflip_prob=dc.tooth_vflip_prob,
        scale_jitter=dc.tooth_scale_jitter,
        color_jitter=dc.tooth_color_jitter,
        brightness=dc.tooth_color_jitter_brightness,
        contrast=dc.tooth_color_jitter_contrast,
    )
    eden_aug = DepthAugmentor(
        rotate_degrees=dc.eden_rotate_degrees,
        hflip_prob=dc.eden_hflip_prob,
        vflip_prob=dc.eden_vflip_prob,
        scale_jitter=dc.eden_scale_jitter,
        color_jitter=dc.eden_color_jitter,
        brightness=dc.eden_color_jitter_brightness,
        contrast=dc.eden_color_jitter_contrast,
    )

    splits_to_load = ["train", "val", "test"] if use_all_splits else ["train"]

    train_dsets, val_dsets = [], []
    split_counts = {}

    for split in splits_to_load:
        sdir = matching_data_dir / split
        if not sdir.is_dir():
            continue
        try:
            # 带 aug（用于训练）
            ds_aug = DepthOnlyDataset(
                data_dir=sdir, labels_csv=labels_csv,
                image_size=dc.image_size,
                tooth_augmentor=tooth_aug,
                eden_augmentor=eden_aug,
                normalizer=normalizer, split=split,
            )
            # 不带 aug（用于验证）— 相同 split 同一批路径，只是 augmentor=None
            ds_clean = DepthOnlyDataset(
                data_dir=sdir, labels_csv=labels_csv,
                image_size=dc.image_size,
                tooth_augmentor=None,    # ← 无增强
                eden_augmentor=None,     # ← 无增强
                normalizer=normalizer, split=split,
            )
            if len(ds_aug) > 0:
                train_dsets.append(ds_aug)
                val_dsets.append(ds_clean)
                split_counts[split] = len(ds_aug)
                print(f"[INFO] Loaded {len(ds_aug)} samples from {split}/")
        except RuntimeError as e:
            print(f"[WARNING] {split}: {e}")

    if not train_dsets:
        raise RuntimeError(f"No valid samples found in {matching_data_dir}")

    def _merge(dsets):
        if len(dsets) == 1:
            return dsets[0]
        merged_ds = ConcatDataset(dsets)
        merged_samples = []
        for ds in dsets:
            merged_samples.extend(ds.samples)

        class _Merged:
            def __init__(self, ds, samples):
                self.dataset = ds
                self.samples = samples
            def __len__(self):        return len(self.dataset)
            def __getitem__(self, i): return self.dataset[i]

        return _Merged(merged_ds, merged_samples)

    train_dataset = _merge(train_dsets)
    val_dataset   = _merge(val_dsets)

    total = len(train_dataset)
    pos   = sum(train_dataset.samples[i]["label"] for i in range(total))
    print(f"[INFO] Total {total} samples  pos={pos} neg={total-pos}  "
          f"({pos/total*100:.1f}% positive)  splits={split_counts}")
    print(f"[INFO] train_dataset: with augmentation")
    print(f"[INFO] val_dataset:   WITHOUT augmentation (clean evaluation)")

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------

def train_one_fold(fold, train_indices, val_indices,
                   train_dataset, val_dataset,
                   cfg, args, output_dir, device):
    """
    FIX: 分别接收 train_dataset（有增强）和 val_dataset（无增强）。
    """
    train_ds = Subset(train_dataset, train_indices)
    val_ds   = Subset(val_dataset,   val_indices)   # ← 无增强的 val

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )

    # 类别平衡
    train_labels = [train_dataset.samples[i]["label"] for i in train_indices]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = max(neg_count / max(pos_count, 1), 1.0)

    # FIX: 用 CLI 传入的 focal_alpha 覆盖 config 里的 0.25
    cfg.train.focal_alpha = args.focal_alpha

    model     = build_depth_only_model(cfg, head_dropout=args.head_dropout).to(device)
    criterion = build_loss(cfg, pos_weight=pos_weight)
    optimizer = build_optimizer(model, cfg)
    scheduler, warmup_scheduler = build_scheduler(optimizer, cfg)

    accum_steps = args.accum_steps
    backbone_lr = cfg.train.learning_rate * cfg.train.backbone_lr_scale

    fold_dir = output_dir / f"fold_{fold + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(
        f"ToothMatchNet.DepthOnly.fold{fold+1}",
        log_file=fold_dir / "train.log",
    )

    eff_bs = cfg.train.batch_size * accum_steps
    logger.info("=" * 65)
    logger.info(f"Fold {fold+1}/{cfg.train.k_folds}  (Depth-Only, v3-fixed)")
    logger.info("=" * 65)
    logger.info(f"Train: {len(train_indices)} (augmented)  "
                f"pos={pos_count} neg={neg_count} pos_weight={pos_weight:.3f}")
    logger.info(f"Val  : {len(val_indices)} (NO augmentation)")
    logger.info(f"Model: {cfg.model.backbone}  1ch-input  "
                f"scale_dim={SCALE_DIM}  params={model.num_parameters:,}")
    logger.info(f"Loss : {cfg.train.loss_type}  "
                f"focal_alpha={args.focal_alpha}  (原0.25→{args.focal_alpha})")
    logger.info(f"LR   : {cfg.train.learning_rate:.2e}  "
                f"backbone×{cfg.train.backbone_lr_scale}={backbone_lr:.2e}")
    logger.info(f"WD   : {cfg.train.weight_decay:.2e}")
    logger.info(f"BS   : {cfg.train.batch_size}  (eff={eff_bs})")
    logger.info(f"head_dropout={args.head_dropout}  (原0.3)")
    logger.info(f"freeze_epochs={cfg.model.freeze_backbone_epochs}  "
                f"(0=不冻结，depth图建议不冻结)")
    logger.info("")

    best_f1, best_epoch, patience_counter = 0.0, 0, 0

    for epoch in range(cfg.train.epochs):

        # Freeze/unfreeze（当 freeze_epochs=0 时跳过）
        if cfg.model.freeze_backbone_epochs > 0:
            if epoch == 0:
                n_frozen, n_train = freeze_backbone(model)
                logger.info(f"[Freeze] backbone frozen ({n_frozen:,} params), "
                            f"trainable={n_train:,}")
            elif epoch == cfg.model.freeze_backbone_epochs:
                n_train = unfreeze_all(model)
                reset_backbone_lr(optimizer, backbone_lr)
                logger.info(f"[Unfreeze] All unfrozen, trainable={n_train:,}, "
                            f"backbone LR reset → {backbone_lr:.2e}")

        # ---- Train ----
        model.train()
        train_loss, train_preds, train_lbls = 0.0, [], []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            tooth_img  = batch["tooth_img"].to(device)
            eden_img   = batch["eden_img"].to(device)
            labels     = batch["label"].to(device)
            scale_feat = batch["scale_feature"].to(device)  # [B, 3]

            logits = model(tooth_img, eden_img, scale_feat)
            loss   = criterion(logits, labels) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_lbls .extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        tp_arr      = np.array(train_preds)
        train_acc   = float(np.mean((tp_arr > 0.5) == np.array(train_lbls)))

        # ---- Validate (no augmentation) ----
        model.eval()
        val_loss, val_preds, val_lbls = 0.0, [], []

        with torch.no_grad():
            for batch in val_loader:
                tooth_img  = batch["tooth_img"].to(device)
                eden_img   = batch["eden_img"].to(device)
                labels     = batch["label"].to(device)
                scale_feat = batch["scale_feature"].to(device)

                logits    = model(tooth_img, eden_img, scale_feat)
                val_loss += criterion(logits, labels).item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_lbls .extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        vp = np.array(val_preds)
        vl = np.array(val_lbls)
        va = float(np.mean((vp > 0.5) == vl))

        tp = int(np.sum((vp > 0.5) & (vl == 1)))
        fp = int(np.sum((vp > 0.5) & (vl == 0)))
        tn = int(np.sum((vp <= 0.5) & (vl == 0)))
        fn = int(np.sum((vp <= 0.5) & (vl == 1)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        v_mean = float(np.mean(vp))
        v_std  = float(np.std(vp))
        t_std  = float(np.std(tp_arr))

        current_lr = optimizer.param_groups[0]["lr"]
        if warmup_scheduler is not None and epoch < cfg.train.warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        # 诊断标志
        if v_std < 0.05:
            diag = "  ← 无区分度！(features collapsed)"
        elif abs(v_mean - 0.5) < 0.02 and v_std < 0.1:
            diag = "  ← 模型在猜 0.5"
        else:
            diag = ""

        logger.info(
            f"Epoch {epoch+1:03d}/{cfg.train.epochs} | LR={current_lr:.2e} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.3f} σ={t_std:.3f} | "
            f"Val Loss={val_loss:.4f} Acc={va:.3f} F1={f1:.4f}"
        )
        logger.info(
            f"         | P={prec:.4f} R={rec:.4f} Spec={spec:.4f} | "
            f"TP={tp} FP={fp} TN={tn} FN={fn}"
        )
        logger.info(
            f"         | val prob μ={v_mean:.4f} σ={v_std:.4f}" + diag
        )

        if f1 > best_f1:
            best_f1, best_epoch, patience_counter = f1, epoch + 1, 0
            save_checkpoint({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1":              best_f1,
                "backbone":             cfg.model.backbone,
                "depth_only":           True,
                "scale_dim":            SCALE_DIM,
            }, fold_dir / "best_model.pth")
            logger.info(f"         ★ Best: F1={best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.train.early_stopping_patience:
            logger.info(f"Early stopping @ epoch {epoch+1}")
            break

    logger.info("")
    logger.info("=" * 65)
    logger.info(f"Fold {fold+1} done. Best F1={best_f1:.4f} @ epoch {best_epoch}")
    logger.info("=" * 65)

    return {
        "fold":          fold + 1,
        "best_f1":       best_f1,
        "best_epoch":    best_epoch,
        "train_samples": len(train_indices),
        "val_samples":   len(val_indices),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = Config()

    cfg.model.backbone               = args.backbone
    cfg.model.freeze_backbone_epochs = args.freeze_epochs
    cfg.train.epochs                 = args.epochs
    cfg.train.batch_size             = args.batch_size
    cfg.train.learning_rate          = args.lr
    cfg.train.weight_decay           = args.weight_decay
    cfg.train.k_folds                = args.k_folds
    cfg.train.use_kfold              = True
    cfg.train.seed                   = args.seed
    cfg.train.device                 = args.device

    set_seed(args.seed)

    project_root = Path(__file__).parent.parent
    output_dir   = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("KFold.DepthOnly.v3", log_file=output_dir / "train.log")
    logger.info("=" * 65)
    logger.info("ToothMatchNet K-Fold Depth-Only  [v3-fixed]")
    logger.info("=" * 65)
    logger.info(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"  backbone      : {args.backbone}")
    logger.info(f"  epochs        : {args.epochs}")
    logger.info(f"  batch_size    : {args.batch_size}  "
                f"(eff={args.batch_size * args.accum_steps})")
    logger.info(f"  lr            : {args.lr:.2e}")
    logger.info(f"  weight_decay  : {args.weight_decay:.2e}")
    logger.info(f"  freeze_epochs : {args.freeze_epochs}  (0=不冻结)")
    logger.info(f"  head_dropout  : {args.head_dropout}  (原0.3)")
    logger.info(f"  focal_alpha   : {args.focal_alpha}  (原0.25)")
    logger.info(f"  scale_dim     : {SCALE_DIM}  (原1，现包含绝对尺寸)")
    logger.info(f"  k_folds       : {args.k_folds}")
    logger.info("")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
    logger.info("")

    matching_data_dir = (Path(args.data_dir) if args.data_dir
                         else project_root / "MatchingData")
    logger.info(f"Data: {matching_data_dir}")

    # FIX: 获取 train/val 两套 dataset
    train_dataset, val_dataset = create_datasets(
        cfg, matching_data_dir, args.use_all_splits)

    all_indices = list(range(len(train_dataset)))
    all_labels  = [train_dataset.samples[i]["label"] for i in all_indices]
    logger.info("")

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True,
                          random_state=args.seed)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(all_indices, all_labels)):
        if args.fold is not None and fold != args.fold:
            continue
        result = train_one_fold(
            fold,
            [all_indices[i] for i in tr_idx],
            [all_indices[i] for i in va_idx],
            train_dataset, val_dataset,   # ← 传两个独立 dataset
            cfg, args, output_dir, device,
        )
        fold_results.append(result)

    logger.info("")
    logger.info("=" * 65)
    logger.info("K-Fold Summary")
    logger.info("=" * 65)
    for r in fold_results:
        logger.info(f"  Fold {r['fold']}: F1={r['best_f1']:.4f} "
                    f"epoch={r['best_epoch']}  "
                    f"train={r['train_samples']} val={r['val_samples']}")
    if fold_results:
        f1s = [r["best_f1"] for r in fold_results]
        logger.info("-" * 65)
        logger.info(f"  Mean ± Std : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        logger.info(f"  Best/Worst : {max(f1s):.4f} / {min(f1s):.4f}")
    logger.info(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 65)

    summary = {
        "config": {
            "backbone":             args.backbone,
            "input_mode":           "depth_only",
            "in_channels":          1,
            "scale_dim":            SCALE_DIM,
            "epochs":               args.epochs,
            "batch_size":           args.batch_size,
            "effective_batch_size": args.batch_size * args.accum_steps,
            "lr":                   args.lr,
            "weight_decay":         args.weight_decay,
            "freeze_epochs":        args.freeze_epochs,
            "head_dropout":         args.head_dropout,
            "focal_alpha":          args.focal_alpha,
            "k_folds":              args.k_folds,
        },
        "fold_results": fold_results,
        "mean_f1": float(np.mean([r["best_f1"] for r in fold_results]))
                   if fold_results else 0.0,
        "std_f1":  float(np.std([r["best_f1"] for r in fold_results]))
                   if fold_results else 0.0,
    }
    out_json = output_dir / "kfold_depth_only_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved → {out_json}")


if __name__ == "__main__":
    main()