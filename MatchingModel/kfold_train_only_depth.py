"""
kfold_train_only_depth.py — K-Fold Training (Depth-Only, v4-Siamese)

v4 核心改变：放弃 cross-attention，改用 Siamese 网络 + 直接特征比较
===========================================================================

为什么 cross-attention 在这里不工作：
  - ResNet50 输出 7×7 feature map → 49 个 token
  - 对 49 个 token 取均值后，所有样本的均值向量几乎相同
  - Cross-attention 参数量 >> 718 个样本能提供的信息量
  - 结果：head 学会输出 logit=0（sigmoid=0.5）使 loss 最小
  - 症状：val prob μ=0.5000 σ=0.0000（精确的 logit=0）

新架构（Siamese-style）：
  tooth_img → Backbone → GlobalAvgPool → [B, feat_dim]
  eden_img  → Backbone → GlobalAvgPool → [B, feat_dim]
                                  ↓
    concat [f_t, f_e, |f_t - f_e|, f_t * f_e, scale_feat]
                                  ↓
                            MLP → logit

  这种方式：
  - 参数量从 51M 降到 ~25M（共享权重）或保持分开
  - 直接比较特征差异，适合小样本匹配任务
  - GlobalAvgPool 避免了 49-token 均值坍塌问题
  - 差值特征 |f_t - f_e| 直接编码"哪里不同"
  - 乘积特征 f_t * f_e 编码"哪里相似"
"""

import os
import sys
import argparse
import json
import random
import csv
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision.transforms.functional as TF
import torchvision.models as tvm
from PIL import Image
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from MatchingModel.config import Config
from MatchingModel.model import _BACKBONE_REGISTRY
from MatchingModel.losses import build_loss
from MatchingModel.utils import get_logger, save_checkpoint, build_scheduler


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_depth_only(depth_path: Path,
                    image_size: Tuple[int, int]) -> Tuple[torch.Tensor, float]:
    target_h, target_w = image_size
    img = Image.open(depth_path).convert("L")
    orig_w, orig_h = img.size

    scale  = min(target_w / orig_w, target_h / orig_h)
    new_w  = int(round(orig_w * scale))
    new_h  = int(round(orig_h * scale))
    img    = img.resize((new_w, new_h), Image.BILINEAR)
    pad_t  = (target_h - new_h) // 2
    pad_l  = (target_w - new_w) // 2
    t      = TF.to_tensor(img)
    t      = F.pad(t, [pad_l, target_w - new_w - pad_l, pad_t, target_h - new_h - pad_t])
    return t, scale


class Normalize1ch:
    def __init__(self, mean=(0.5,), std=(0.5,)):
        self.mean = torch.tensor(mean).view(1, 1, 1)
        self.std  = torch.tensor(std).view(1, 1, 1)
    def __call__(self, x):
        return (x - self.mean) / self.std


class DepthAugmentor:
    def __init__(self, rotate_degrees=0., hflip_prob=0., vflip_prob=0.,
                 scale_jitter=0., color_jitter=False, brightness=0., contrast=0.):
        self.rotate_degrees = rotate_degrees
        self.hflip_prob     = hflip_prob
        self.vflip_prob     = vflip_prob
        self.scale_jitter   = scale_jitter
        self.color_jitter   = color_jitter
        self.brightness     = brightness
        self.contrast       = contrast

    def __call__(self, img):
        H, W = img.shape[-2], img.shape[-1]
        if self.scale_jitter > 0:
            s = random.uniform(1. - self.scale_jitter, 1. + self.scale_jitter)
            nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
            img = F.interpolate(img.unsqueeze(0), (nh, nw),
                                mode="bilinear", align_corners=False).squeeze(0)
            if s > 1.:
                img = img[:, (nh-H)//2:(nh-H)//2+H, (nw-W)//2:(nw-W)//2+W]
            else:
                pt, pl = (H-nh)//2, (W-nw)//2
                img = F.pad(img, [pl, W-nw-pl, pt, H-nh-pt])
        if self.rotate_degrees > 0:
            a = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            img = TF.rotate(img, a, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
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
                img = torch.clamp((img - .5) * c + .5, 0, 1)
        return img


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SCALE_DIM = 3


class DepthOnlyDataset(Dataset):
    SUFFIXES = {"eden_depth": "_eden_depth.png", "tooth_depth": "_tooth_depth.png"}

    def __init__(self, data_dir, labels_csv, image_size=(224, 224),
                 tooth_aug=None, eden_aug=None, normalizer=None, split="train"):
        self.data_dir   = Path(data_dir)
        self.image_size = image_size
        self.tooth_aug  = tooth_aug
        self.eden_aug   = eden_aug
        self.normalizer = normalizer
        self.split      = split
        self.samples    = self._load(labels_csv)

    @staticmethod
    def _find(sdir, sfx):
        m = sorted(p for p in sdir.iterdir() if p.is_file() and p.name.endswith(sfx))
        return m[0] if m else None

    def _load(self, csv_path):
        samples, has_split = [], None
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if all(v is None or str(v).strip() == "" for v in row.values()):
                    continue
                row = {k.strip().lstrip("\ufeff"): v for k, v in row.items()}
                if has_split is None:
                    has_split = "split" in row
                if has_split and row.get("split", "").strip() != self.split:
                    continue
                sid   = row["sample_id"].strip()
                label = int(row["label"].strip())
                sdir  = self.data_dir / sid
                if not sdir.is_dir():
                    continue
                paths, miss = {}, []
                for k, sfx in self.SUFFIXES.items():
                    f2 = self._find(sdir, sfx)
                    if f2 is None: miss.append(sfx)
                    else:          paths[k] = f2
                if miss:
                    print(f"[Dataset] skip {sid}: missing {miss}")
                    continue
                samples.append({"sample_id": sid, "label": label, **paths})
        if not samples:
            raise RuntimeError(f"No samples for split='{self.split}' in {self.data_dir}")
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tooth, ts = load_depth_only(s["tooth_depth"], self.image_size)
        eden,  es = load_depth_only(s["eden_depth"],  self.image_size)
        if self.tooth_aug:  tooth = self.tooth_aug(tooth)
        if self.eden_aug:   eden  = self.eden_aug(eden)
        if self.normalizer:
            tooth = self.normalizer(tooth)
            eden  = self.normalizer(eden)
        eps = 1e-6
        sf  = torch.tensor([
            float(np.log(max(ts, eps))),
            float(np.log(max(es, eps))),
            float(np.log(ts / max(es, eps))),
        ], dtype=torch.float32)
        return {
            "tooth_img":     tooth,
            "eden_img":      eden,
            "scale_feature": sf,
            "label":         torch.tensor(s["label"], dtype=torch.float32),
            "sample_id":     s["sample_id"],
        }

    @property
    def labels(self): return [s["label"] for s in self.samples]


# ---------------------------------------------------------------------------
# Siamese model (replaces ToothMatchNet + cross-attention)
# ---------------------------------------------------------------------------

def _make_backbone(name: str, pretrained: bool):
    """Returns (backbone_module, feat_dim) where backbone outputs [B, C, H, W]."""
    reg = _BACKBONE_REGISTRY
    if name not in reg:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {list(reg)}")
    factory_fn, weights_enum, feat_dim = reg[name]
    weights = weights_enum if pretrained else None
    full    = factory_fn(weights=weights)

    if "resnet" in name:
        layers = [full.conv1, full.bn1, full.relu, full.maxpool,
                  full.layer1, full.layer2, full.layer3, full.layer4]
        bb = nn.Sequential(*layers)
        # Replace conv1 for 1-channel input
        old = full.conv1
        bb[0] = nn.Conv2d(1, old.out_channels, old.kernel_size,
                          old.stride, old.padding, bias=False)
        nn.init.kaiming_normal_(bb[0].weight, mode="fan_out", nonlinearity="relu")
    else:  # convnext
        bb = full.features
        stem = bb[0][0]
        new_stem = nn.Conv2d(1, stem.out_channels, stem.kernel_size,
                             stem.stride, stem.padding,
                             bias=stem.bias is not None)
        nn.init.kaiming_normal_(new_stem.weight, mode="fan_out", nonlinearity="relu")
        if new_stem.bias is not None:
            nn.init.zeros_(new_stem.bias)
        bb[0][0] = new_stem

    return bb, feat_dim


class SiameseMatchNet(nn.Module):
    """
    Siamese 匹配网络，替代 ToothMatchNet + cross-attention。

    特征比较方式（4路拼接）：
        [f_t, f_e, |f_t - f_e|, f_t * f_e]  形状 [B, 4*feat_dim]

    直接编码了"两个特征哪里相同、哪里不同"，
    比 cross-attention 均值池化更能保留匹配信号。

    共享 vs 分开权重：
        shared=True  — 两分支共享 backbone（适合对称匹配）
        shared=False — 两分支独立（适合非对称匹配，如牙列 vs 牙颌）
                       牙列方向固定(U形)，牙颌方向任意，适合 shared=False
    """

    def __init__(self, backbone_name: str, pretrained: bool = True,
                 shared: bool = False,
                 head_hidden: Tuple[int, ...] = (512, 128),
                 head_dropout: float = 0.1,
                 scale_dim: int = SCALE_DIM):
        super().__init__()
        self.shared = shared

        self.tooth_bb, feat_dim = _make_backbone(backbone_name, pretrained)
        if shared:
            self.eden_bb = self.tooth_bb
        else:
            self.eden_bb, _ = _make_backbone(backbone_name, pretrained)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # 4路特征 + scale_feature
        in_dim = feat_dim * 4 + scale_dim
        layers = []
        prev = in_dim
        for h in head_hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(head_dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, backbone, x):
        feat = backbone(x)                    # [B, C, H, W]
        return self.pool(feat).flatten(1)     # [B, C]

    def forward(self, tooth, eden, scale_feat=None):
        ft = self.encode(self.tooth_bb, tooth)  # [B, feat_dim]
        fe = self.encode(self.eden_bb,  eden)

        diff = (ft - fe).abs()
        prod = ft * fe
        x    = torch.cat([ft, fe, diff, prod], dim=1)   # [B, 4*feat_dim]

        if scale_feat is None:
            scale_feat = torch.zeros(x.shape[0], SCALE_DIM,
                                     device=x.device, dtype=x.dtype)
        x = torch.cat([x, scale_feat], dim=1)            # [B, 4*feat_dim + scale_dim]
        return self.head(x).squeeze(-1)                  # [B]

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Optimizer for siamese model (两组 LR)
# ---------------------------------------------------------------------------

def build_siamese_optimizer(model: SiameseMatchNet, cfg, backbone_lr_scale: float):
    tc = cfg.train
    bb_params  = list(model.tooth_bb.parameters())
    if not model.shared:
        bb_params += list(model.eden_bb.parameters())
    bb_ids     = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]

    bb_lr = tc.learning_rate * backbone_lr_scale
    groups = [
        {"params": bb_params,   "lr": bb_lr,            "name": "backbone"},
        {"params": head_params, "lr": tc.learning_rate,  "name": "head"},
    ]
    return torch.optim.AdamW(groups, weight_decay=tc.weight_decay)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def create_datasets(cfg, data_root: Path, use_all_splits: bool = False):
    dc  = cfg.data
    norm = Normalize1ch()
    csv_path = data_root / "labels.csv"

    tooth_aug = DepthAugmentor(
        rotate_degrees=dc.tooth_rotate_degrees, hflip_prob=dc.tooth_hflip_prob,
        vflip_prob=dc.tooth_vflip_prob, scale_jitter=dc.tooth_scale_jitter,
        color_jitter=dc.tooth_color_jitter,
        brightness=dc.tooth_color_jitter_brightness,
        contrast=dc.tooth_color_jitter_contrast,
    )
    eden_aug = DepthAugmentor(
        rotate_degrees=dc.eden_rotate_degrees, hflip_prob=dc.eden_hflip_prob,
        vflip_prob=dc.eden_vflip_prob, scale_jitter=dc.eden_scale_jitter,
        color_jitter=dc.eden_color_jitter,
        brightness=dc.eden_color_jitter_brightness,
        contrast=dc.eden_color_jitter_contrast,
    )

    splits = ["train", "val", "test"] if use_all_splits else ["train"]
    tr_dsets, va_dsets, counts = [], [], {}

    for sp in splits:
        sdir = data_root / sp
        if not sdir.is_dir(): continue
        try:
            ds_aug   = DepthOnlyDataset(sdir, csv_path, dc.image_size,
                                        tooth_aug, eden_aug, norm, sp)
            ds_clean = DepthOnlyDataset(sdir, csv_path, dc.image_size,
                                        None, None, norm, sp)
            if len(ds_aug) > 0:
                tr_dsets.append(ds_aug)
                va_dsets.append(ds_clean)
                counts[sp] = len(ds_aug)
                print(f"[INFO] {sp}: {len(ds_aug)} samples")
        except RuntimeError as e:
            print(f"[WARNING] {sp}: {e}")

    if not tr_dsets:
        raise RuntimeError(f"No data found in {data_root}")

    def merge(dsets):
        if len(dsets) == 1: return dsets[0]
        merged = ConcatDataset(dsets)
        all_samples = []
        for d in dsets: all_samples.extend(d.samples)
        class W:
            def __init__(self, d, s): self.dataset=d; self.samples=s
            def __len__(self): return len(self.dataset)
            def __getitem__(self, i): return self.dataset[i]
        return W(merged, all_samples)

    tr = merge(tr_dsets)
    va = merge(va_dsets)
    n  = len(tr)
    pos = sum(tr.samples[i]["label"] for i in range(n))
    print(f"[INFO] Total {n}  pos={pos} neg={n-pos}  ({pos/n*100:.1f}% pos)  {counts}")
    return tr, va


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="K-Fold Siamese Depth-Only (v4)")

    p.add_argument("--k-folds",            type=int,   default=5)
    p.add_argument("--fold",               type=int,   default=None)
    p.add_argument("--backbone",           type=str,   default="resnet50",
                   choices=list(_BACKBONE_REGISTRY.keys()))

    # 共享权重 vs 独立权重
    # 牙列(U形，固定方向) vs 牙颌(任意方向) → 非对称匹配 → 推荐不共享
    p.add_argument("--shared-backbone",    action="store_true",
                   help="两分支共享 backbone 权重（默认不共享，因为牙列/牙颌非对称）")
    p.add_argument("--no-pretrained",      action="store_true")

    p.add_argument("--epochs",             type=int,   default=100)
    p.add_argument("--batch-size",         type=int,   default=16,
                   help="Siamese 网络建议用更大 batch（≥16），使特征对比更稳定")
    p.add_argument("--accum-steps",        type=int,   default=1)
    p.add_argument("--lr",                 type=float, default=3e-4)
    p.add_argument("--backbone-lr-scale",  type=float, default=0.1,
                   help="backbone LR = lr * scale（Siamese 建议 0.1，让 head 先学会比较）")
    p.add_argument("--weight-decay",       type=float, default=1e-2)
    p.add_argument("--head-dropout",       type=float, default=0.1)
    p.add_argument("--head-hidden",        type=int,   nargs="+", default=[512, 128],
                   help="head 隐藏层大小（默认 512 128）")
    p.add_argument("--focal-alpha",        type=float, default=0.5)
    p.add_argument("--warmup-epochs",      type=int,   default=5)

    p.add_argument("--data-dir",           type=str,   default=None)
    p.add_argument("--use-all-splits",     action="store_true")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--output-dir",         type=str,
                   default="MatchingCheckpoints/KFold_Siamese")
    p.add_argument("--device",             type=str,   default="cuda")

    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------

def train_one_fold(fold, tr_idx, va_idx, tr_ds, va_ds, cfg, args, out_dir, device):
    tr_loader = DataLoader(
        Subset(tr_ds, tr_idx), batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor, drop_last=True,
    )
    va_loader = DataLoader(
        Subset(va_ds, va_idx), batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )

    tr_labels = [tr_ds.samples[i]["label"] for i in tr_idx]
    pos_n = sum(tr_labels);  neg_n = len(tr_labels) - pos_n
    pos_w = max(neg_n / max(pos_n, 1), 1.0)

    cfg.train.focal_alpha = args.focal_alpha

    model = SiameseMatchNet(
        backbone_name = args.backbone,
        pretrained    = not args.no_pretrained,
        shared        = args.shared_backbone,
        head_hidden   = tuple(args.head_hidden),
        head_dropout  = args.head_dropout,
    ).to(device)

    criterion = build_loss(cfg, pos_weight=pos_w)
    optimizer = build_siamese_optimizer(model, cfg, args.backbone_lr_scale)
    scheduler, warmup_sched = build_scheduler(optimizer, cfg)

    fold_dir = out_dir / f"fold_{fold+1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"Siamese.fold{fold+1}", log_file=fold_dir / "train.log")

    bb_lr = cfg.train.learning_rate * args.backbone_lr_scale
    logger.info("=" * 65)
    logger.info(f"Fold {fold+1}/{cfg.train.k_folds}  [Siamese v4]")
    logger.info("=" * 65)
    logger.info(f"Train: {len(tr_idx)}  pos={pos_n} neg={neg_n} pw={pos_w:.3f}")
    logger.info(f"Val  : {len(va_idx)} (no augmentation)")
    logger.info(f"Model: {args.backbone}  shared={args.shared_backbone}  "
                f"params={model.num_parameters:,}")
    logger.info(f"Head : {args.head_hidden}  dropout={args.head_dropout}")
    logger.info(f"LR   : head={cfg.train.learning_rate:.2e}  "
                f"backbone={bb_lr:.2e}  (scale={args.backbone_lr_scale})")
    logger.info(f"BS   : {cfg.train.batch_size}  WD={cfg.train.weight_decay:.2e}")
    logger.info("")

    best_f1, best_ep, patience = 0., 0, 0
    accum = args.accum_steps

    for epoch in range(cfg.train.epochs):
        model.train()
        tr_loss, tr_preds, tr_lbls = 0., [], []
        optimizer.zero_grad()

        for step, batch in enumerate(tr_loader):
            tooth = batch["tooth_img"].to(device)
            eden  = batch["eden_img"].to(device)
            lbl   = batch["label"].to(device)
            sf    = batch["scale_feature"].to(device)

            logits = model(tooth, eden, sf)
            loss   = criterion(logits, lbl) / accum
            loss.backward()

            if (step+1) % accum == 0 or (step+1) == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            tr_loss  += loss.item() * accum
            tr_preds .extend(torch.sigmoid(logits).detach().cpu().numpy())
            tr_lbls  .extend(lbl.cpu().numpy())

        tr_loss /= len(tr_loader)
        tp_arr   = np.array(tr_preds)
        tr_acc   = float(np.mean((tp_arr > .5) == np.array(tr_lbls)))
        t_std    = float(np.std(tp_arr))

        model.eval()
        va_loss, va_preds, va_lbls = 0., [], []
        with torch.no_grad():
            for batch in va_loader:
                tooth = batch["tooth_img"].to(device)
                eden  = batch["eden_img"].to(device)
                lbl   = batch["label"].to(device)
                sf    = batch["scale_feature"].to(device)
                out   = model(tooth, eden, sf)
                va_loss += criterion(out, lbl).item()
                va_preds.extend(torch.sigmoid(out).cpu().numpy())
                va_lbls .extend(lbl.cpu().numpy())

        va_loss /= len(va_loader)
        vp = np.array(va_preds);  vl = np.array(va_lbls)
        va_acc = float(np.mean((vp > .5) == vl))
        tp = int(np.sum((vp>.5)&(vl==1))); fp = int(np.sum((vp>.5)&(vl==0)))
        tn = int(np.sum((vp<=.5)&(vl==0))); fn = int(np.sum((vp<=.5)&(vl==1)))
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.
        v_mu = float(np.mean(vp));  v_std = float(np.std(vp))

        cur_lr = optimizer.param_groups[1]["lr"]  # head LR
        if warmup_sched is not None and epoch < cfg.train.warmup_epochs:
            warmup_sched.step()
        elif scheduler is not None:
            scheduler.step()

        flag = ""
        if v_std < 0.05:
            flag = "  ← 无区分度"
        elif tp == len(vl[vl==1]) and tn == 0:
            flag = "  ← 全猜正"
        elif tn == len(vl[vl==0]) and tp == 0:
            flag = "  ← 全猜负"

        logger.info(
            f"Ep {epoch+1:03d}/{cfg.train.epochs} | LR={cur_lr:.2e} | "
            f"Tr Loss={tr_loss:.4f} Acc={tr_acc:.3f} σ={t_std:.3f} | "
            f"Val Loss={va_loss:.4f} Acc={va_acc:.3f} F1={f1:.4f}"
        )
        logger.info(
            f"          | P={prec:.4f} R={rec:.4f} Spec={spec:.4f} | "
            f"TP={tp} FP={fp} TN={tn} FN={fn}"
        )
        logger.info(f"          | val prob μ={v_mu:.4f} σ={v_std:.4f}{flag}")

        if f1 > best_f1:
            best_f1, best_ep, patience = f1, epoch+1, 0
            save_checkpoint({
                "epoch": epoch+1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1, "backbone": args.backbone,
                "shared": args.shared_backbone,
            }, fold_dir / "best_model.pth")
            logger.info(f"          ★ Best F1={best_f1:.4f}")
        else:
            patience += 1

        if patience >= cfg.train.early_stopping_patience:
            logger.info(f"Early stopping @ epoch {epoch+1}")
            break

    logger.info("=" * 65)
    logger.info(f"Fold {fold+1} done. Best F1={best_f1:.4f} @ epoch {best_ep}")
    logger.info("=" * 65)

    return {"fold": fold+1, "best_f1": best_f1, "best_epoch": best_ep,
            "train_samples": len(tr_idx), "val_samples": len(va_idx)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = Config()

    cfg.model.backbone          = args.backbone
    cfg.model.pretrained        = not args.no_pretrained
    cfg.train.epochs            = args.epochs
    cfg.train.batch_size        = args.batch_size
    cfg.train.learning_rate     = args.lr
    cfg.train.weight_decay      = args.weight_decay
    cfg.train.backbone_lr_scale = args.backbone_lr_scale
    cfg.train.warmup_epochs     = args.warmup_epochs
    cfg.train.k_folds           = args.k_folds
    cfg.train.seed              = args.seed

    set_seed(args.seed)

    proj_root = Path(__file__).parent.parent
    out_dir   = proj_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("Siamese.KFold", log_file=out_dir / "train.log")
    logger.info("=" * 65)
    logger.info("ToothMatchNet Siamese K-Fold  [v4]")
    logger.info("=" * 65)
    logger.info(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"  backbone         : {args.backbone}")
    logger.info(f"  shared_backbone  : {args.shared_backbone}")
    logger.info(f"  pretrained       : {not args.no_pretrained}")
    logger.info(f"  head_hidden      : {args.head_hidden}")
    logger.info(f"  head_dropout     : {args.head_dropout}")
    logger.info(f"  epochs           : {args.epochs}")
    logger.info(f"  batch_size       : {args.batch_size}  "
                f"(eff={args.batch_size * args.accum_steps})")
    logger.info(f"  lr               : {args.lr:.2e}  "
                f"backbone={args.lr * args.backbone_lr_scale:.2e}")
    logger.info(f"  weight_decay     : {args.weight_decay:.2e}")
    logger.info(f"  focal_alpha      : {args.focal_alpha}")
    logger.info(f"  warmup_epochs    : {args.warmup_epochs}")
    logger.info(f"  k_folds          : {args.k_folds}")
    logger.info("")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
    logger.info("")

    data_root = Path(args.data_dir) if args.data_dir else proj_root / "MatchingData"
    logger.info(f"Data: {data_root}")

    tr_ds, va_ds = create_datasets(cfg, data_root, args.use_all_splits)
    all_idx    = list(range(len(tr_ds)))
    all_labels = [tr_ds.samples[i]["label"] for i in all_idx]
    logger.info("")

    skf     = StratifiedKFold(n_splits=args.k_folds, shuffle=True,
                              random_state=args.seed)
    results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(all_idx, all_labels)):
        if args.fold is not None and fold != args.fold:
            continue
        r = train_one_fold(
            fold,
            [all_idx[i] for i in tr_idx],
            [all_idx[i] for i in va_idx],
            tr_ds, va_ds, cfg, args, out_dir, device,
        )
        results.append(r)

    logger.info("")
    logger.info("=" * 65)
    logger.info("K-Fold Summary  [Siamese v4]")
    logger.info("=" * 65)
    for r in results:
        logger.info(f"  Fold {r['fold']}: F1={r['best_f1']:.4f}  "
                    f"ep={r['best_epoch']}  "
                    f"tr={r['train_samples']} va={r['val_samples']}")
    if results:
        f1s = [r["best_f1"] for r in results]
        logger.info("-" * 65)
        logger.info(f"  Mean ± Std : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        logger.info(f"  Best/Worst : {max(f1s):.4f} / {min(f1s):.4f}")
    logger.info(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")

    summary = {
        "config": {
            "backbone": args.backbone, "shared": args.shared_backbone,
            "pretrained": not args.no_pretrained,
            "head_hidden": args.head_hidden, "head_dropout": args.head_dropout,
            "lr": args.lr, "backbone_lr_scale": args.backbone_lr_scale,
            "batch_size": args.batch_size, "weight_decay": args.weight_decay,
            "focal_alpha": args.focal_alpha, "k_folds": args.k_folds,
        },
        "fold_results": results,
        "mean_f1": float(np.mean([r["best_f1"] for r in results])) if results else 0.,
        "std_f1":  float(np.std([r["best_f1"] for r in results])) if results else 0.,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()