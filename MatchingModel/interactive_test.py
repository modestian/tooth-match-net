"""
interactive_test.py - Interactive testing script for ToothMatchNet.

Features:
    1. Select single sample or batch test mode
    2. Choose which K-Fold model to use
    3. Interactive sample selection for single mode
"""

import sys
import csv
from pathlib import Path
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.cuda.amp import autocast

from config import CFG, TEST_DIR, LABELS_CSV, CHECKPOINT_DIR
from model import build_model
from dataset import load_branch, build_normalizer
from utils import get_logger, compute_metrics


def get_logger_local():
    return get_logger("ToothMatchNet.test")


def list_test_samples(test_dir: Path) -> List[str]:
    """List all sample directories in test_dir."""
    if not test_dir.exists():
        return []
    samples = sorted([
        d.name for d in test_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return samples


def list_available_models() -> Dict[str, Path]:
    """List available model checkpoints."""
    models = {}
    
    kfold_dir = CHECKPOINT_DIR / "KFold"
    if kfold_dir.exists():
        for fold_dir in sorted(kfold_dir.iterdir()):
            if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
                for ckpt_name in ["best_model.pth", "best.pth"]:
                    best_path = fold_dir / ckpt_name
                    if best_path.exists():
                        models[fold_dir.name] = best_path
                        break
    
    best_link = CHECKPOINT_DIR / "best.pth"
    if best_link.exists():
        models["best.pth (default)"] = best_link
    
    for pth in sorted(CHECKPOINT_DIR.glob("*.pth")):
        if pth.name not in models:
            models[pth.name] = pth
    
    return models


def list_kfold_models() -> List[Path]:
    """List all K-Fold model checkpoints for ensemble."""
    kfold_dir = CHECKPOINT_DIR / "KFold"
    if not kfold_dir.exists():
        return []
    
    models = []
    for fold_dir in sorted(kfold_dir.iterdir()):
        if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
            for ckpt_name in ["best_model.pth", "best.pth"]:
                best_path = fold_dir / ckpt_name
                if best_path.exists():
                    models.append(best_path)
                    break
    return models


def get_backbone_from_checkpoint(checkpoint_path: Path) -> str:
    """Get backbone config from checkpoint or kfold_summary.json."""
    default_backbone = CFG.model.backbone
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "backbone" in ckpt:
        return ckpt["backbone"]
    
    parent_dir = checkpoint_path.parent
    if parent_dir.name.startswith("fold_"):
        kfold_dir = parent_dir.parent
    else:
        kfold_dir = parent_dir
    
    summary_path = kfold_dir / "kfold_summary.json"
    if summary_path.exists():
        import json
        with open(summary_path) as f:
            summary = json.load(f)
        if "config" in summary and "backbone" in summary["config"]:
            return summary["config"]["backbone"]
    
    return default_backbone


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    logger = get_logger_local()
    
    backbone = get_backbone_from_checkpoint(checkpoint_path)
    logger.info(f"Using backbone: {backbone}")
    
    from config import Config, ModelConfig
    cfg = Config()
    cfg.model.backbone = backbone
    
    model = build_model(cfg).to(device)
    model.eval()
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    
    epoch = ckpt.get("epoch", "N/A")
    val_f1 = ckpt.get("best_f1", ckpt.get("val_f1", "N/A"))
    if isinstance(val_f1, float):
        logger.info(f"  Checkpoint epoch: {epoch}, Best F1: {val_f1:.4f}")
    else:
        logger.info(f"  Checkpoint epoch: {epoch}, Best F1: {val_f1}")
    
    return model


def load_ensemble_models(device: torch.device) -> List[torch.nn.Module]:
    """Load all K-Fold models for ensemble prediction."""
    logger = get_logger_local()
    
    kfold_paths = list_kfold_models()
    if not kfold_paths:
        logger.error("No K-Fold models found for ensemble!")
        return []
    
    backbone = get_backbone_from_checkpoint(kfold_paths[0])
    logger.info(f"Loading {len(kfold_paths)} models for ensemble (backbone: {backbone})")
    
    from config import Config
    cfg = Config()
    cfg.model.backbone = backbone
    
    models = []
    for path in kfold_paths:
        model = build_model(cfg).to(device)
        model.eval()
        
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        models.append(model)
        
        fold_name = path.parent.name
        f1 = ckpt.get("best_f1", "N/A")
        if isinstance(f1, float):
            logger.info(f"  Loaded {fold_name}: F1={f1:.4f}")
        else:
            logger.info(f"  Loaded {fold_name}")
    
    return models


def _find_file_by_suffix(sample_dir: Path, suffix: str) -> Path:
    """Find file ending with suffix in sample_dir."""
    matches = sorted([
        p for p in sample_dir.iterdir()
        if p.is_file() and p.name.endswith(suffix)
    ])
    if len(matches) == 0:
        raise FileNotFoundError(f"No file ending with '{suffix}' found in {sample_dir}")
    return matches[0]


def predict_single(model: torch.nn.Module,
                   sample_dir: Path,
                   device: torch.device,
                   use_tta: bool = False) -> float:
    """Predict single sample with single model, return probability."""
    normalizer = build_normalizer(CFG)
    image_size = CFG.data.image_size
    
    sfxs = {
        "tooth_depth":  "_tooth_depth.png",
        "tooth_normal": "_tooth_normal.png",
        "eden_depth":   "_eden_depth.png",
        "eden_normal":  "_eden_normal.png",
    }
    
    paths = {k: _find_file_by_suffix(sample_dir, sfx) for k, sfx in sfxs.items()}
    
    tooth_img, _ = load_branch(paths["tooth_depth"], paths["tooth_normal"], image_size)
    eden_img, _ = load_branch(paths["eden_depth"], paths["eden_normal"], image_size)
    
    tooth_img = normalizer(tooth_img).unsqueeze(0).to(device)
    eden_img = normalizer(eden_img).unsqueeze(0).to(device)
    
    if use_tta:
        probs = []
        for angle in CFG.infer.tta_rotations:
            import torchvision.transforms.functional as TF
            eden_rot = TF.rotate(eden_img, angle,
                                 interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            with torch.no_grad():
                with autocast(enabled=CFG.infer.use_amp):
                    logit = model(tooth_img, eden_rot)
                probs.append(torch.sigmoid(logit).item())
            
            if CFG.infer.tta_hflip:
                eden_flip = TF.hflip(eden_rot)
                with torch.no_grad():
                    with autocast(enabled=CFG.infer.use_amp):
                        logit = model(tooth_img, eden_flip)
                    probs.append(torch.sigmoid(logit).item())
        return float(np.mean(probs))
    else:
        with torch.no_grad():
            with autocast(enabled=CFG.infer.use_amp):
                logit = model(tooth_img, eden_img)
            return torch.sigmoid(logit).item()


def predict_ensemble(models: List[torch.nn.Module],
                     sample_dir: Path,
                     device: torch.device,
                     use_tta: bool = False) -> float:
    """Predict single sample with ensemble of models (soft voting), return probability."""
    all_probs = []
    for model in models:
        prob = predict_single(model, sample_dir, device, use_tta)
        all_probs.append(prob)
    
    return float(np.mean(all_probs))


def run_single_test(models: List[torch.nn.Module],
                    test_dir: Path,
                    sample_id: str,
                    device: torch.device,
                    threshold: float = 0.5,
                    use_tta: bool = False,
                    gt_label: Optional[int] = None):
    """Run single sample test."""
    logger = get_logger_local()
    
    sample_dir = test_dir / sample_id
    if not sample_dir.exists():
        logger.error(f"Sample directory not found: {sample_dir}")
        return
    
    if len(models) > 1:
        prob = predict_ensemble(models, sample_dir, device, use_tta)
    else:
        prob = predict_single(models[0], sample_dir, device, use_tta)
    pred = int(prob >= threshold)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Single Sample Test Result")
    logger.info("=" * 60)
    logger.info(f"  Sample ID   : {sample_id}")
    logger.info(f"  Probability : {prob:.4f}")
    logger.info(f"  Prediction  : {'MATCH (1)' if pred == 1 else 'NO MATCH (0)'}")
    logger.info(f"  Threshold   : {threshold}")
    if len(models) > 1:
        logger.info(f"  Mode        : Ensemble ({len(models)} models)")
    
    if gt_label is not None:
        correct = "Correct" if pred == gt_label else "WRONG"
        logger.info(f"  Ground Truth: {gt_label}")
        logger.info(f"  Result      : {correct}")
    logger.info("=" * 60)


def run_batch_test(models: List[torch.nn.Module],
                   test_dir: Path,
                   device: torch.device,
                   threshold: float = 0.5,
                   use_tta: bool = False,
                   labels_csv: Optional[Path] = None,
                   output_csv: Optional[Path] = None):
    """Run batch test on all samples in test_dir."""
    logger = get_logger_local()
    
    samples = list_test_samples(test_dir)
    if not samples:
        logger.warning(f"No samples found in {test_dir}")
        return
    
    logger.info(f"Found {len(samples)} samples in {test_dir}")
    if len(models) > 1:
        logger.info(f"Using ensemble of {len(models)} models")
    
    gt_map: Dict[str, int] = {}
    if labels_csv and labels_csv.exists():
        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_map[row["sample_id"].strip()] = int(row["label"].strip())
        logger.info(f"Loaded {len(gt_map)} ground truth labels")
    
    results = []
    all_probs = []
    all_preds = []
    all_gts = []
    
    for i, sid in enumerate(samples, 1):
        sample_dir = test_dir / sid
        try:
            if len(models) > 1:
                prob = predict_ensemble(models, sample_dir, device, use_tta)
            else:
                prob = predict_single(models[0], sample_dir, device, use_tta)
            pred = int(prob >= threshold)
            gt = gt_map.get(sid)
            
            result = {
                "sample_id": sid,
                "probability": round(prob, 6),
                "prediction": pred,
                "label": gt if gt is not None else "",
                "correct": (int(pred == gt) if gt is not None else "")
            }
            results.append(result)
            all_probs.append(prob)
            all_preds.append(pred)
            if gt is not None:
                all_gts.append(gt)
            
            status = ""
            if gt is not None:
                status = f" | GT={gt} {'OK' if pred == gt else 'WRONG'}"
            logger.info(f"  [{i}/{len(samples)}] {sid}: prob={prob:.4f}, pred={pred}{status}")
            
        except Exception as e:
            logger.warning(f"  [{i}/{len(samples)}] {sid}: ERROR - {e}")
    
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"\nResults saved to: {output_csv}")
    
    if all_gts and len(all_gts) == len(all_preds):
        metrics = compute_metrics(
            np.array(all_preds), np.array(all_gts), np.array(all_probs), threshold
        )
        logger.info("")
        logger.info("=" * 60)
        logger.info("  Batch Test Results")
        logger.info("=" * 60)
        logger.info(f"  Total samples: {len(results)}")
        logger.info(f"  Accuracy      : {metrics['accuracy']:.4f}")
        logger.info(f"  Precision     : {metrics['precision']:.4f}")
        logger.info(f"  Recall        : {metrics['recall']:.4f}")
        logger.info(f"  F1 Score      : {metrics['f1']:.4f}")
        logger.info(f"  Specificity   : {metrics['specificity']:.4f}")
        logger.info(f"  TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
        logger.info("=" * 60)


def print_menu(title: str, options: List[str]) -> int:
    """Print menu and get user selection."""
    print("")
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("-" * 50)
    
    while True:
        try:
            choice = input("  Please select [1-{}]: ".format(len(options)))
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"  Invalid input. Please enter 1-{len(options)}")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def main():
    logger = get_logger_local()
    
    print("")
    print("=" * 60)
    print("  ToothMatchNet Interactive Test")
    print("=" * 60)
    
    test_dir = TEST_DIR
    samples = list_test_samples(test_dir)
    
    if not samples:
        logger.warning(f"No test samples found in {test_dir}")
        logger.info("Please add test data to the test directory first.")
        logger.info("Expected structure: test/sample_xxxx/*_tooth_depth.png, etc.")
        return
    
    kfold_models = list_kfold_models()
    single_models = list_available_models()
    
    if not single_models and not kfold_models:
        logger.error("No model checkpoints found!")
        return
    
    model_options = []
    if kfold_models:
        model_options.append("Ensemble (All 5 Folds) - Recommended")
    model_options.extend(list(single_models.keys()))
    
    model_idx = print_menu("Select Model", model_options)
    
    use_ensemble = (kfold_models and model_idx == 0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if use_ensemble:
        logger.info("Selected: Ensemble (All 5 Folds)")
        models = load_ensemble_models(device)
        if not models:
            logger.error("Failed to load ensemble models!")
            return
    else:
        actual_idx = model_idx - 1 if kfold_models else model_idx
        selected_model_name = list(single_models.keys())[actual_idx]
        checkpoint_path = single_models[selected_model_name]
        logger.info(f"Selected model: {selected_model_name}")
        models = [load_model(checkpoint_path, device)]
    
    mode_options = ["Single Sample Test", "Batch Test (All Samples)"]
    mode_idx = print_menu("Select Test Mode", mode_options)
    
    use_tta = False
    tta_options = ["No TTA (faster)", "Use TTA (more accurate)"]
    tta_idx = print_menu("Test-Time Augmentation", tta_options)
    use_tta = (tta_idx == 1)
    
    threshold = CFG.infer.threshold
    threshold_input = input(f"  Classification threshold [{threshold}]: ").strip()
    if threshold_input:
        try:
            threshold = float(threshold_input)
        except ValueError:
            logger.warning(f"Invalid threshold, using default: {threshold}")
    
    if mode_idx == 0:
        sample_options = samples
        if len(samples) > 10:
            print("")
            print(f"  Found {len(samples)} samples. Showing first 20:")
            for i, s in enumerate(samples[:20], 1):
                print(f"    {i}. {s}")
            if len(samples) > 20:
                print(f"    ... and {len(samples) - 20} more")
            
            sample_input = input("  Enter sample ID or number: ").strip()
            if sample_input.isdigit():
                idx = int(sample_input) - 1
                if 0 <= idx < len(samples):
                    selected_sample = samples[idx]
                else:
                    logger.error("Invalid sample number")
                    return
            else:
                selected_sample = sample_input
        else:
            sample_idx = print_menu("Select Sample", samples)
            selected_sample = samples[sample_idx]
        
        gt_map: Dict[str, int] = {}
        if LABELS_CSV.exists():
            with open(LABELS_CSV) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gt_map[row["sample_id"].strip()] = int(row["label"].strip())
        gt_label = gt_map.get(selected_sample)
        
        run_single_test(models, test_dir, selected_sample, device, threshold, use_tta, gt_label)
    
    else:
        output_csv = CHECKPOINT_DIR / "test_results.csv"
        output_input = input(f"  Output CSV path [{output_csv}]: ").strip()
        if output_input:
            output_csv = Path(output_input)
        
        run_batch_test(models, test_dir, device, threshold, use_tta, LABELS_CSV, output_csv)


if __name__ == "__main__":
    main()
