"""
kfold_train.py — K-Fold Cross Validation Training for ToothMatchNet

Usage:
    python kfold_train.py --k-folds 5 --epochs 100
    python kfold_train.py --backbone resnet50 --lr 1e-4

Features:
    - K-Fold交叉验证，自动划分训练/验证集
    - 默认只在 train 数据上进行交叉验证，确保数据分布一致
    - 每折独立训练，保存最佳模型
    - 汇总K折平均性能指标
    - 支持ConvNeXt和ResNet backbone

Data Loading:
    默认从 MatchingData/train 目录加载数据进行 K-Fold 交叉验证。
    如需使用所有数据，可通过 --use-all-splits 参数合并 train/val/test。
"""

import os
import sys
import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from MatchingModel.config import CFG, ModelConfig, DataConfig, TrainConfig, Config
from MatchingModel.model import build_model
from MatchingModel.dataset import ToothMatchDataset, BranchAugmentor, Normalize4ch
from MatchingModel.losses import build_loss
from MatchingModel.utils import (
    build_optimizer, build_scheduler, get_logger,
    save_checkpoint, load_checkpoint
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="K-Fold Cross Validation Training")
    
    # K-Fold参数
    parser.add_argument("--k-folds", type=int, default=5, help="K折数")
    parser.add_argument("--fold", type=int, default=None, help="只训练特定折(0到k_folds-1)")
    
    # 模型参数
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["convnext_tiny", "convnext_small", "convnext_base",
                                "resnet18", "resnet34", "resnet50", "resnet101"],
                        help="Backbone网络")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="每折训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="权重衰减")
    parser.add_argument("--freeze-epochs", type=int, default=20, help="冻结backbone的epoch数")
    
    # 数据参数
    parser.add_argument("--data-dir", type=str, default=None, 
                        help="数据目录 (默认: 项目根目录/MatchingData/train)")
    parser.add_argument("--use-all-splits", action="store_true",
                        help="合并 train/val/test 所有数据进行交叉验证")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 其他
    parser.add_argument("--output-dir", type=str, default="MatchingCheckpoints/KFold", 
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(cfg: Config, matching_data_dir: Path, use_all_splits: bool = False):
    """
    创建数据集用于K-Fold划分。
    
    Args:
        cfg: 配置对象
        matching_data_dir: MatchingData 目录路径 (包含 train/val/test 子目录和 labels.csv)
        use_all_splits: 是否合并所有 splits (train/val/test)，默认只用 train
    
    Returns:
        数据集
    """
    dc = cfg.data
    
    # 数据增强
    tooth_aug = BranchAugmentor(
        rotate_degrees=dc.tooth_rotate_degrees,
        hflip_prob=dc.tooth_hflip_prob,
        vflip_prob=dc.tooth_vflip_prob,
        scale_jitter=dc.tooth_scale_jitter,
        color_jitter=dc.tooth_color_jitter,
        brightness=dc.tooth_color_jitter_brightness,
        contrast=dc.tooth_color_jitter_contrast,
    )
    
    eden_aug = BranchAugmentor(
        rotate_degrees=dc.eden_rotate_degrees,
        hflip_prob=dc.eden_hflip_prob,
        vflip_prob=dc.eden_vflip_prob,
        scale_jitter=dc.eden_scale_jitter,
        color_jitter=dc.eden_color_jitter,
        brightness=dc.eden_color_jitter_brightness,
        contrast=dc.eden_color_jitter_contrast,
    )
    
    normalizer = Normalize4ch(
        mean=dc.normalize_mean,
        std=dc.normalize_std,
    )
    
    labels_csv = matching_data_dir / "labels.csv"
    
    if not use_all_splits:
        # 默认：只加载 train split
        train_dir = matching_data_dir / "train"
        dataset = ToothMatchDataset(
            data_dir=train_dir,
            labels_csv=labels_csv,
            image_size=dc.image_size,
            tooth_augmentor=tooth_aug,
            eden_augmentor=eden_aug,
            normalizer=normalizer,
            split="train",
        )
        print(f"[INFO] Loaded {len(dataset)} samples from train directory (use_all_splits=False)")
        return dataset
    
    # 合并所有 splits
    all_datasets = []
    split_counts = {}
    
    for split in ["train", "val", "test"]:
        split_dir = matching_data_dir / split
        if split_dir.is_dir():
            try:
                dataset = ToothMatchDataset(
                    data_dir=split_dir,
                    labels_csv=labels_csv,
                    image_size=dc.image_size,
                    tooth_augmentor=tooth_aug,
                    eden_augmentor=eden_aug,
                    normalizer=normalizer,
                    split=split,
                )
                if len(dataset) > 0:
                    all_datasets.append(dataset)
                    split_counts[split] = len(dataset)
                    print(f"[INFO] Loaded {len(dataset)} samples from {split} directory")
            except RuntimeError as e:
                print(f"[WARNING] Could not load {split} split: {e}")
    
    if not all_datasets:
        raise RuntimeError(
            f"No valid samples found in {matching_data_dir}. "
            f"Please check that train/val/test directories exist with valid samples."
        )
    
    # 合并所有数据集
    if len(all_datasets) == 1:
        merged_dataset = all_datasets[0]
    else:
        merged_dataset = torch.utils.data.ConcatDataset(all_datasets)
    
    # 为合并后的数据集添加 samples 属性（用于获取标签）
    merged_samples = []
    for ds in all_datasets:
        merged_samples.extend(ds.samples)
    
    # 创建一个包装类来保存合并后的 samples
    class MergedDataset:
        def __init__(self, dataset, samples):
            self.dataset = dataset
            self.samples = samples
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            return self.dataset[idx]
    
    merged_dataset = MergedDataset(merged_dataset, merged_samples)
    
    total_samples = len(merged_dataset)
    print(f"[INFO] Total samples loaded: {total_samples}")
    print(f"[INFO] Split distribution: {split_counts}")
    
    return merged_dataset


def train_one_fold(
    fold: int,
    train_indices: List[int],
    val_indices: List[int],
    full_dataset: ToothMatchDataset,
    cfg: Config,
    output_dir: Path,
    device: torch.device
) -> Dict[str, float]:
    """训练一折"""
    
    # 创建子数据集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    
    # 计算pos_weight
    train_labels = [full_dataset.samples[i]["label"] for i in train_indices]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # 创建模型
    model = build_model(cfg).to(device)
    
    # 损失函数
    criterion = build_loss(cfg, pos_weight=pos_weight)
    
    # 优化器
    optimizer = build_optimizer(model, cfg)
    
    # 学习率调度器
    scheduler, warmup_scheduler = build_scheduler(optimizer, cfg)
    
    # 日志 - 使用唯一的 logger 名称，避免重复
    fold_dir = output_dir / f"fold_{fold + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"ToothMatchNet.fold{fold+1}", log_file=fold_dir / "train.log")
    
    # 输出完整的 fold 信息
    logger.info("=" * 60)
    logger.info(f"Training Fold {fold + 1}/{cfg.train.k_folds}")
    logger.info("=" * 60)
    logger.info(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    logger.info(f"Train pos/neg: {pos_count}/{neg_count}, pos_weight: {pos_weight:.3f}")
    logger.info(f"Model: {cfg.model.backbone}")
    logger.info(f"Params: {model.num_parameters:,} total, {model.num_trainable_parameters:,} trainable")
    logger.info(f"Loss: {cfg.train.loss_type}")
    logger.info(f"Learning rate: {cfg.train.learning_rate:.2e}")
    logger.info(f"Batch size: {cfg.train.batch_size}")
    logger.info(f"Epochs: {cfg.train.epochs}")
    logger.info("")
    
    # 训练循环
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(cfg.train.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []
        
        for batch in train_loader:
            tooth_img = batch["tooth_img"].to(device)
            eden_img = batch["eden_img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(tooth_img, eden_img)
            loss = criterion(logits, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = np.mean((np.array(train_preds) > 0.5) == np.array(train_labels_list))
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                tooth_img = batch["tooth_img"].to(device)
                eden_img = batch["eden_img"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(tooth_img, eden_img)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.array(val_preds)
        val_labels_arr = np.array(val_labels_list)
        
        # 计算指标
        val_acc = np.mean((val_preds > 0.5) == val_labels_arr)
        
        # 混淆矩阵
        tp = int(np.sum((val_preds > 0.5) & (val_labels_arr == 1)))
        fp = int(np.sum((val_preds > 0.5) & (val_labels_arr == 0)))
        tn = int(np.sum((val_preds <= 0.5) & (val_labels_arr == 0)))
        fn = int(np.sum((val_preds <= 0.5) & (val_labels_arr == 1)))
        
        # 计算更多指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 概率分布统计
        prob_mean = np.mean(val_preds)
        prob_std = np.std(val_preds)
        
        # 学习率更新
        current_lr = optimizer.param_groups[0]["lr"]
        if warmup_scheduler is not None and epoch < cfg.train.warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()
        
        # 完整日志输出
        logger.info(
            f"Epoch {epoch+1:03d}/{cfg.train.epochs} | "
            f"LR={current_lr:.2e} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={f1:.4f}"
        )
        logger.info(
            f"         | "
            f"P={precision:.4f} R={recall:.4f} Spec={specificity:.4f} | "
            f"TP={tp} FP={fp} TN={tn} FN={fn}"
        )
        logger.info(
            f"         | "
            f"prob_mean={prob_mean:.4f} prob_std={prob_std:.4f}"
            + ("  ← 模型输出无区分度！" if prob_std < 0.05 else "")
        )
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            patience_counter = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "backbone": cfg.model.backbone,
            }
            save_checkpoint(checkpoint, fold_dir / "best_model.pth")
            logger.info(f"         ★ New best checkpoint saved: F1={best_f1:.4f}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= cfg.train.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
            break
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Fold {fold + 1} Complete")
    logger.info(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    logger.info("=" * 60)
    
    return {
        "fold": fold + 1,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
    }


def main():
    """主函数"""
    args = parse_args()
    
    # 设置配置
    cfg = Config()
    
    # 模型配置
    cfg.model.backbone = args.backbone
    cfg.model.freeze_backbone_epochs = args.freeze_epochs
    
    # 训练配置
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.k_folds = args.k_folds
    cfg.train.use_kfold = True
    cfg.train.seed = args.seed
    cfg.train.device = args.device
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 输出目录 - 使用项目根目录下的绝对路径
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "MatchingCheckpoints" / "KFold"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建全局日志文件
    log_path = output_dir / "train.log"
    logger = get_logger("ToothMatchNet.KFoldMain", log_file=log_path)
    
    # 输出完整的配置信息
    logger.info("=" * 60)
    logger.info("ToothMatchNet K-Fold Cross Validation Training")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Backbone: {args.backbone}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr:.2e}")
    logger.info(f"  Weight decay: {args.weight_decay:.2e}")
    logger.info(f"  Freeze epochs: {args.freeze_epochs}")
    logger.info(f"  K-Folds: {args.k_folds}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Use all splits: {args.use_all_splits}")
    logger.info("")
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("")
    
    # 数据目录 - 指向 MatchingData 目录（包含 train/val/test 子目录）
    if args.data_dir:
        matching_data_dir = Path(args.data_dir)
    else:
        matching_data_dir = project_root / "MatchingData"
    
    logger.info(f"Data directory: {matching_data_dir}")
    
    # 创建数据集
    if args.use_all_splits:
        logger.info("Loading dataset from all splits (train/val/test)...")
    else:
        logger.info("Loading dataset from train split only...")
    
    full_dataset = create_dataset(cfg, matching_data_dir, use_all_splits=args.use_all_splits)
    logger.info(f"Total samples: {len(full_dataset)}")
    
    # 获取所有样本索引
    all_indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i]["label"] for i in all_indices]
    
    # K-Fold划分（分层抽样）
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # 记录每折结果
    fold_results = []
    
    # 训练每折
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, labels)):
        if args.fold is not None and fold != args.fold:
            continue
        
        train_indices = [all_indices[i] for i in train_idx]
        val_indices = [all_indices[i] for i in val_idx]
        
        result = train_one_fold(
            fold, train_indices, val_indices,
            full_dataset, cfg, output_dir, device
        )
        fold_results.append(result)
    
    # 汇总结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("K-Fold Cross Validation Summary")
    logger.info("=" * 60)
    
    for r in fold_results:
        logger.info(
            f"Fold {r['fold']}: Best F1 = {r['best_f1']:.4f} "
            f"(Epoch {r['best_epoch']}, Train={r['train_samples']}, Val={r['val_samples']})"
        )
    
    avg_f1 = np.mean([r['best_f1'] for r in fold_results])
    std_f1 = np.std([r['best_f1'] for r in fold_results])
    max_f1 = max([r['best_f1'] for r in fold_results])
    min_f1 = min([r['best_f1'] for r in fold_results])
    
    logger.info("-" * 60)
    logger.info(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"Best F1:    {max_f1:.4f}")
    logger.info(f"Worst F1:   {min_f1:.4f}")
    logger.info("-" * 60)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 保存汇总结果
    summary = {
        "config": {
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "k_folds": args.k_folds,
        },
        "fold_results": fold_results,
        "average_f1": float(avg_f1),
        "std_f1": float(std_f1),
    }
    
    with open(output_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {output_dir / 'kfold_summary.json'}")


if __name__ == "__main__":
    main()
