"""
train.py — Training script for ToothMatchNet.

Usage:
    python train.py                          # use default CFG
    python train.py --epochs 50 --lr 3e-4   # override specific params
    python train.py --resume                 # resume from best checkpoint
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure MatchingModel is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from config import CFG, CHECKPOINT_DIR
from model import build_model
from dataset import build_dataloaders
from losses import build_loss
from utils import (
    set_seed, get_logger, AverageMeter, compute_metrics,
    build_optimizer, build_scheduler, CheckpointManager,
    EarlyStopping, model_summary, count_samples_per_class,
    tensor_to_device, format_time,
)


# ---------------------------------------------------------------------------
# Parse CLI arguments (allow overriding CFG without editing config.py)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ToothMatchNet")
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batch-size",    type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--backbone",      type=str,   default=None,
                   choices=["convnext_tiny", "convnext_small", "convnext_base"])
    p.add_argument("--loss",          type=str,   default=None,
                   choices=["bce", "focal", "bce_focal"])
    p.add_argument("--no-amp",        action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume",        action="store_true",
                   help="Resume from best checkpoint in CHECKPOINT_DIR")
    p.add_argument("--seed",          type=int,   default=None)
    p.add_argument("--device",        type=str,   default=None)
    return p.parse_args()


def apply_args(cfg, args):
    """Apply CLI overrides to cfg (mutates cfg in-place)."""
    if args.epochs        is not None: cfg.train.epochs         = args.epochs
    if args.batch_size    is not None: cfg.train.batch_size      = args.batch_size
    if args.lr            is not None: cfg.train.learning_rate   = args.lr
    if args.backbone      is not None: cfg.model.backbone        = args.backbone
    if args.loss          is not None: cfg.train.loss_type       = args.loss
    if args.no_amp:                    cfg.train.use_amp         = False
    if args.no_pretrained:             cfg.model.pretrained      = False
    if args.seed          is not None: cfg.train.seed            = args.seed
    if args.device        is not None: cfg.train.device          = args.device
    return cfg


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, cfg, logger, epoch, global_step):
    model.train()
    tc = cfg.train

    loss_meter = AverageMeter("loss")
    all_preds, all_labels, all_probs = [], [], []

    t0 = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch         = tensor_to_device(batch, device)
        tooth_img     = batch["tooth_img"]
        eden_img      = batch["eden_img"]
        labels        = batch["label"]
        scale_feature = batch.get("scale_feature", None)  # [B, 1] or None

        # Forward pass (AMP)
        with autocast("cuda", enabled=tc.use_amp):
            logits = model(tooth_img, eden_img, scale_feature)
            loss   = criterion(logits, labels)
            if tc.accumulation_steps > 1:
                loss = loss / tc.accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % tc.accumulation_steps == 0 or (step + 1) == len(loader):
            if tc.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        # Metrics
        loss_val = loss.item() * tc.accumulation_steps
        loss_meter.update(loss_val, labels.size(0))

        # logits shape 可能是 [B, 1] 或 [B]，统一 squeeze 成 [B]
        probs_np = torch.sigmoid(logits).detach().cpu().numpy().ravel()   # (B,)
        preds_np = (probs_np >= 0.5).astype(int)                          # (B,)
        all_probs.extend(probs_np.tolist())
        all_preds.extend(preds_np.tolist())
        all_labels.extend(labels.cpu().numpy().ravel().astype(int).tolist())

        if (step + 1) % tc.log_every_n_steps == 0:
            elapsed = format_time(time.time() - t0)
            logger.info(
                f"[Train] Epoch {epoch:03d} | Step {step+1}/{len(loader)} "
                f"| Loss {loss_meter.avg:.4f} | Elapsed {elapsed}"
            )

    metrics = compute_metrics(
        np.array(all_preds), np.array(all_labels), np.array(all_probs)
    )
    metrics["loss"] = loss_meter.avg
    return metrics, global_step


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    tc = cfg.train

    loss_meter = AverageMeter("val_loss")
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch     = tensor_to_device(batch, device)
        tooth_img = batch["tooth_img"]
        eden_img  = batch["eden_img"]
        labels    = batch["label"]

        with autocast("cuda", enabled=tc.use_amp):
            logits = model(tooth_img, eden_img)
            loss   = criterion(logits, labels)

        loss_meter.update(loss.item(), labels.size(0))

        # 同样 squeeze 成 1D
        probs_np = torch.sigmoid(logits).cpu().numpy().ravel()   # (B,)
        preds_np = (probs_np >= 0.5).astype(int)                  # (B,)
        all_probs.extend(probs_np.tolist())
        all_preds.extend(preds_np.tolist())
        all_labels.extend(labels.cpu().numpy().ravel().astype(int).tolist())

    probs_arr = np.array(all_probs)
    metrics = compute_metrics(
        np.array(all_preds), np.array(all_labels), probs_arr
    )
    metrics["loss"]       = loss_meter.avg
    metrics["prob_mean"]  = float(probs_arr.mean())
    metrics["prob_std"]   = float(probs_arr.std())
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = apply_args(CFG, args)
    tc   = cfg.train

    # ---- Setup ----
    set_seed(tc.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CHECKPOINT_DIR / "train.log"
    logger   = get_logger("ToothMatchNet.train", log_file=log_path)

    device = torch.device(
        tc.device if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")

    # ---- Data ----
    logger.info("Building dataloaders …")
    train_loader, val_loader, _ = build_dataloaders(cfg)
    train_info = count_samples_per_class(train_loader.dataset)
    val_info   = count_samples_per_class(val_loader.dataset)
    logger.info(f"Train: {train_info}")
    logger.info(f"Val  : {val_info}")

    # ---- Model ----
    logger.info(f"Building model: {cfg.model.backbone} …")
    model = build_model(cfg).to(device)

    if tc.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    logger.info("\n" + model_summary(model))

    # ---- Loss ----
    pos_weight = (train_loader.dataset.pos_weight
                  if tc.pos_weight is None else tc.pos_weight)
    criterion  = build_loss(cfg, pos_weight=pos_weight)
    criterion  = criterion.to(device)
    logger.info(f"Loss: {criterion.__class__.__name__} | pos_weight={pos_weight:.4f}")

    # ---- Optimizer & Scheduler ----
    raw_model  = model.module if isinstance(model, nn.DataParallel) else model
    optimizer  = build_optimizer(raw_model, cfg)
    scheduler, warmup_scheduler = build_scheduler(optimizer, cfg,
                                                   steps_per_epoch=len(train_loader))
    scaler     = GradScaler("cuda", enabled=tc.use_amp)

    # ---- Checkpoint manager ----
    ckpt_mgr     = CheckpointManager(CHECKPOINT_DIR, top_k=tc.keep_top_k_checkpoints)
    early_stop   = EarlyStopping(patience=tc.early_stopping_patience)
    start_epoch  = 1
    global_step  = 0

    # ---- Optional: resume ----
    if args.resume:
        ckpt = ckpt_mgr.load_best(device=str(device))
        if ckpt:
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
            logger.info(f"Resumed from epoch {start_epoch - 1} "
                        f"(val_f1={ckpt.get('val_f1', 'N/A')})")
        else:
            logger.warning("No checkpoint found for resume. Starting fresh.")

    # ---- TensorBoard ----
    writer = None
    if tc.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(CHECKPOINT_DIR / "tb_logs"))
            logger.info("TensorBoard writer created.")
        except ImportError:
            logger.warning("TensorBoard not installed. Skipping TB logging.")

    # ---- Backbone freeze/unfreeze strategy ----
    # Model now has SEPARATE tooth_encoder / eden_encoder (each with .adapter + .backbone)
    # We freeze only the .backbone part of each encoder; .adapter is always trainable.
    freeze_epochs = getattr(cfg.model, "freeze_backbone_epochs", 0)
    backbone_frozen = False

    def freeze_backbone(m):
        """Freeze ConvNeXt backbone of both branches; keep adapters trainable."""
        frozen_params = 0
        for enc in [m.tooth_encoder, m.eden_encoder]:
            for p in enc.backbone.parameters():
                p.requires_grad = False
                frozen_params += p.numel()
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        logger.info(
            f"  [Freeze] ConvNeXt backbone frozen ({frozen_params:,} params frozen, "
            f"{trainable:,} trainable). Training: adapter + fusion + head for "
            f"{freeze_epochs} epochs."
        )

    def unfreeze_backbone(m):
        """Unfreeze ConvNeXt backbone of both branches for end-to-end fine-tuning."""
        for enc in [m.tooth_encoder, m.eden_encoder]:
            for p in enc.backbone.parameters():
                p.requires_grad = True
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        logger.info(
            f"  [Unfreeze] Full backbone unfrozen — end-to-end fine-tuning "
            f"({total:,} trainable params)."
        )

    if freeze_epochs > 0:
        freeze_backbone(raw_model)
        backbone_frozen = True

    # ---- Training loop ----
    logger.info(f"Starting training for {tc.epochs} epochs …")
    best_val_f1 = 0.0
    train_start = time.time()

    for epoch in range(start_epoch, tc.epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after freeze_epochs
        if backbone_frozen and epoch > freeze_epochs:
            unfreeze_backbone(raw_model)
            backbone_frozen = False
            # Reset optimizer LR after unfreezing
            for pg in optimizer.param_groups:
                if pg.get("name") == "backbone":
                    pg["lr"] = tc.learning_rate * tc.backbone_lr_scale
            logger.info(f"  [LR reset] backbone LR → {tc.learning_rate * tc.backbone_lr_scale:.2e}")

        # Train
        train_metrics, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg, logger, epoch, global_step
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, cfg)

        # Scheduler step（必须在 optimizer.step() 之后调用）
        if warmup_scheduler is not None and epoch <= tc.warmup_epochs:
            warmup_scheduler.step()
        else:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["f1"])
            else:
                scheduler.step()

        # Current LR
        current_lr = optimizer.param_groups[-1]["lr"]

        epoch_time = format_time(time.time() - epoch_start)
        # Format AUC / AP nicely (handle None / nan)
        def _fmt(v):
            if v is None: return "N/A"
            try:
                return "N/A" if (v != v) else f"{v:.4f}"   # nan check
            except Exception:
                return "N/A"

        # 如果 AUC 还是 nan，打印 probs 分布帮助排查
        val_auc_str = _fmt(val_metrics.get('auc'))
        val_ap_str  = _fmt(val_metrics.get('avg_precision'))
        trn_auc_str = _fmt(train_metrics.get('auc'))

        p_mean = val_metrics.get('prob_mean', float('nan'))
        p_std  = val_metrics.get('prob_std',  float('nan'))
        logger.info(
            f"\n{'='*65}\n"
            f"  Epoch {epoch:03d}/{tc.epochs:03d}  [{epoch_time}]  LR={current_lr:.2e}\n"
            f"  Train | Loss={train_metrics['loss']:.4f} "
            f"Acc={train_metrics['accuracy']:.4f} F1={train_metrics['f1']:.4f} "
            f"AUC={trn_auc_str}\n"
            f"  Val   | Loss={val_metrics['loss']:.4f}  "
            f"Acc={val_metrics['accuracy']:.4f}  F1={val_metrics['f1']:.4f}  "
            f"AUC={val_auc_str}  AP={val_ap_str}\n"
            f"  Val   | P={val_metrics['precision']:.4f} "
            f"R={val_metrics['recall']:.4f}  "
            f"Spec={val_metrics['specificity']:.4f}  "
            f"TP={val_metrics['tp']} FP={val_metrics['fp']} "
            f"TN={val_metrics['tn']} FN={val_metrics['fn']}\n"
            f"  Val   | prob_mean={p_mean:.4f}  prob_std={p_std:.4f}"
            f"{'  ← 模型输出无区分度！' if p_std < 0.05 else ''}\n"
            f"{'='*65}"
        )

        # TensorBoard
        if writer:
            for k, v in train_metrics.items():
                if isinstance(v, float):
                    writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        # Checkpoint
        val_f1  = val_metrics["f1"]
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1

        if epoch % tc.save_every_n_epochs == 0 or is_best:
            state = {
                "epoch":               epoch,
                "global_step":         global_step,
                "model_state_dict":    raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":              val_f1,
                "val_metrics":         val_metrics,
                "train_metrics":       train_metrics,
                "cfg":                 cfg,
            }
            ckpt_path = ckpt_mgr.save(state, val_f1, epoch, is_best=is_best)
            if is_best:
                logger.info(f"  ★ New best checkpoint saved: {ckpt_path.name} "
                            f"(val_f1={val_f1:.4f})")

        # Early stopping
        if early_stop.step(val_f1):
            logger.info(
                f"Early stopping triggered (no improvement for "
                f"{tc.early_stopping_patience} epochs)."
            )
            break

    total_time = format_time(time.time() - train_start)
    logger.info(f"\nTraining complete in {total_time} | Best val F1: {best_val_f1:.4f}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
