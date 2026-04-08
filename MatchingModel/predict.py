"""
predict.py — Inference script for ToothMatchNet.

Supports:
    1. Single sample prediction       (4 image files)
    2. Batch prediction from a folder (structured like train/val/test)
    3. Test-time augmentation (TTA)   for the eden branch (rotation + flip)

Usage examples:
    # Single sample
    python predict.py \\
        --sample-id sample_001 \\
        --data-dir MatchingData/test

    # Batch (whole folder → results.csv)
    python predict.py \\
        --data-dir MatchingData/test \\
        --labels-csv MatchingData/labels.csv \\
        --output results.csv

    # Specify checkpoint explicitly
    python predict.py \\
        --checkpoint MatchingCheckpoints/best.pth \\
        --data-dir MatchingData/test \\
        --output results.csv \\
        --tta
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast

from config import CFG, CHECKPOINT_DIR, TEST_DIR, LABELS_CSV
from model import build_model
from dataset import (
    ToothMatchDataset, load_branch, Normalize4ch,
    BranchAugmentor, build_normalizer,
)
from utils import get_logger, compute_metrics, tensor_to_device


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ToothMatchNet Inference")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to .pth checkpoint (default: best.pth in CHECKPOINT_DIR)")
    p.add_argument("--data-dir",    type=str, default=str(TEST_DIR))
    p.add_argument("--labels-csv",  type=str, default=None,
                   help="Optional: path to labels.csv for metric evaluation")
    p.add_argument("--sample-id",   type=str, default=None,
                   help="Run on a single sample_id")
    p.add_argument("--output",      type=str, default="results.csv",
                   help="Output CSV path for batch predictions")
    p.add_argument("--threshold",   type=float, default=None)
    p.add_argument("--tta",         action="store_true",
                   help="Enable test-time augmentation")
    p.add_argument("--device",      type=str,  default=None)
    p.add_argument("--batch-size",  type=int,  default=None)
    p.add_argument("--no-amp",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Optional[str], cfg,
               device: torch.device) -> torch.nn.Module:
    """Load model and checkpoint weights."""
    logger = get_logger("ToothMatchNet.predict")

    model = build_model(cfg).to(device)
    model.eval()

    if checkpoint_path is None:
        best_link = CHECKPOINT_DIR / "best.pth"
        if best_link.exists():
            checkpoint_path = str(best_link)
        else:
            pths = sorted(CHECKPOINT_DIR.glob("*.pth"),
                          key=lambda p: p.stat().st_mtime)
            if pths:
                checkpoint_path = str(pths[-1])

    if checkpoint_path is None:
        logger.warning("No checkpoint found. Using random weights (for testing only).")
    else:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"  Checkpoint epoch : {ckpt.get('epoch', 'N/A')}")
        logger.info(f"  Best val F1      : {ckpt.get('val_f1', 'N/A')}")

    return model


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

def tta_rotate_eden(eden_img: torch.Tensor,
                    angle: float) -> torch.Tensor:
    """Rotate the eden branch tensor by angle degrees."""
    return TF.rotate(eden_img, angle,
                     interpolation=TF.InterpolationMode.BILINEAR, fill=0)


def tta_hflip_eden(eden_img: torch.Tensor) -> torch.Tensor:
    return TF.hflip(eden_img)


@torch.no_grad()
def predict_with_tta(model: torch.nn.Module,
                     tooth_img: torch.Tensor,
                     eden_img: torch.Tensor,
                     tta_rotations: Tuple[float, ...] = (0, 90, 180, 270),
                     tta_hflip: bool = True,
                     use_amp: bool = True) -> float:
    """
    Average predictions over TTA variants (rotations + optional hflip).
    Returns mean sigmoid probability.
    """
    variants: List[torch.Tensor] = []

    for angle in tta_rotations:
        variants.append(tta_rotate_eden(eden_img, angle))
        if tta_hflip:
            variants.append(tta_hflip_eden(tta_rotate_eden(eden_img, angle)))

    probs = []
    for eden_v in variants:
        with autocast(enabled=use_amp):
            logit = model(tooth_img, eden_v)
        prob = torch.sigmoid(logit).item()
        probs.append(prob)

    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Single sample inference
# ---------------------------------------------------------------------------

def _find_file_by_suffix(sample_dir: Path, suffix: str) -> Path:
    """
    Find the unique file in sample_dir whose name ends with `suffix`.

    - The prefix before the suffix can be anything (e.g. 'scanA', 'nA', 'sc').
    - Matching is done purely on the suffix string (case-sensitive).
    - If MULTIPLE files match, a warning is printed and the lexicographically
      first one is used (deterministic across runs).
    - Raises FileNotFoundError if no matching file is found.
    """
    matches = sorted([
        p for p in sample_dir.iterdir()
        if p.is_file() and p.name.endswith(suffix)
    ])
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file ending with '{suffix}' found in {sample_dir}"
        )
    if len(matches) > 1:
        import logging
        logging.getLogger("ToothMatchNet.predict").warning(
            f"Multiple files match suffix '{suffix}' in {sample_dir}:\n"
            + "\n".join(f"    {m.name}" for m in matches)
            + f"\n  → using '{matches[0].name}' (lexicographic first)"
        )
    return matches[0]


def predict_single(model: torch.nn.Module,
                   data_dir: Path,
                   sample_id: str,
                   cfg,
                   device: torch.device,
                   threshold: float = 0.5,
                   use_tta: bool = False) -> Dict:
    """
    Run inference on a single sample.
    The sample sub-directory is  data_dir / sample_id /
    Files are located by their suffix (prefix before the suffix is arbitrary).
    Returns dict with sample_id, probability, prediction.
    """
    normalizer = build_normalizer(cfg)
    image_size = cfg.data.image_size

    # Each sample lives in its own sub-directory
    sample_dir = data_dir / sample_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Locate the four images by suffix (prefix is arbitrary)
    sfxs = {
        "tooth_depth":  "_tooth_depth.png",
        "tooth_normal": "_tooth_normal.png",
        "eden_depth":   "_eden_depth.png",
        "eden_normal":  "_eden_normal.png",
    }
    paths = {k: _find_file_by_suffix(sample_dir, sfx) for k, sfx in sfxs.items()}

    # Load
    tooth_img, _ = load_branch(paths["tooth_depth"], paths["tooth_normal"], image_size)
    eden_img, _  = load_branch(paths["eden_depth"],  paths["eden_normal"],  image_size)

    # Normalise
    tooth_img = normalizer(tooth_img)
    eden_img  = normalizer(eden_img)

    # Add batch dim, move to device
    tooth_img = tooth_img.unsqueeze(0).to(device)
    eden_img  = eden_img.unsqueeze(0).to(device)

    ic = cfg.infer

    if use_tta:
        prob = predict_with_tta(model, tooth_img, eden_img,
                                tta_rotations=ic.tta_rotations,
                                tta_hflip=ic.tta_hflip,
                                use_amp=ic.use_amp)
    else:
        with torch.no_grad():
            with autocast(enabled=ic.use_amp):
                logit = model(tooth_img, eden_img)
            prob = torch.sigmoid(logit).item()

    pred = int(prob >= threshold)
    return {"sample_id": sample_id, "probability": prob, "prediction": pred}


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(model: torch.nn.Module,
                  data_dir: Path,
                  labels_csv: Optional[Path],
                  cfg,
                  device: torch.device,
                  threshold: float = 0.5,
                  use_tta: bool = False,
                  output_path: Optional[Path] = None) -> List[Dict]:
    """
    Run inference on all samples found by scanning data_dir for image files.
    If labels_csv is provided, compute and print evaluation metrics.
    Writes results to output_path (CSV).
    """
    logger = get_logger("ToothMatchNet.predict")
    ic     = cfg.infer

    # Discover sample IDs from data_dir by listing sub-directories
    sample_ids = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    logger.info(f"Found {len(sample_ids)} sample sub-directories in {data_dir}")

    if not sample_ids:
        logger.warning("No sample sub-directories found in data_dir.")
        return []

    # Load ground truth labels if available
    gt_map: Dict[str, int] = {}
    if labels_csv and labels_csv.exists():
        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_map[row["sample_id"].strip()] = int(row["label"].strip())

    normalizer = build_normalizer(cfg)
    image_size = cfg.data.image_size

    results  = []
    all_prob = []
    all_pred = []
    all_gt   = []

    sfxs = {
        "tooth_depth":  "_tooth_depth.png",
        "tooth_normal": "_tooth_normal.png",
        "eden_depth":   "_eden_depth.png",
        "eden_normal":  "_eden_normal.png",
    }

    for sid in sample_ids:
        sample_dir = data_dir / sid
        # Locate each image by suffix; skip sample if any are missing
        paths = {}
        missing = []
        for key, sfx in sfxs.items():
            try:
                paths[key] = _find_file_by_suffix(sample_dir, sfx)
            except FileNotFoundError:
                missing.append(f"*{sfx}")
        if missing:
            logger.warning(f"  Skipping {sid}: missing {missing} in {sample_dir}")
            continue

        tooth_img, _ = load_branch(paths["tooth_depth"], paths["tooth_normal"], image_size)
        eden_img, _  = load_branch(paths["eden_depth"],  paths["eden_normal"],  image_size)
        tooth_img = normalizer(tooth_img).unsqueeze(0).to(device)
        eden_img  = normalizer(eden_img).unsqueeze(0).to(device)

        if use_tta:
            prob = predict_with_tta(model, tooth_img, eden_img,
                                    tta_rotations=ic.tta_rotations,
                                    tta_hflip=ic.tta_hflip,
                                    use_amp=ic.use_amp)
        else:
            with autocast(enabled=ic.use_amp):
                logit = model(tooth_img, eden_img)
            prob = torch.sigmoid(logit).item()

        pred  = int(prob >= threshold)
        label = gt_map.get(sid, None)

        row = {
            "sample_id":   sid,
            "probability": round(prob, 6),
            "prediction":  pred,
            "label":       label if label is not None else "",
            "correct":     (int(pred == label) if label is not None else ""),
        }
        results.append(row)
        all_prob.append(prob)
        all_pred.append(pred)
        if label is not None:
            all_gt.append(label)

        logger.info(
            f"  {sid} | prob={prob:.4f} | pred={pred}"
            + (f" | gt={label}" if label is not None else "")
        )

    # Write CSV
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"\nResults written to: {output_path}")

    # Metrics
    if all_gt and len(all_gt) == len(all_pred):
        metrics = compute_metrics(
            np.array(all_pred), np.array(all_gt), np.array(all_prob), threshold
        )
        logger.info("\n" + "=" * 50)
        logger.info("  Evaluation Metrics")
        logger.info("=" * 50)
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k:<20}: {v:.4f}")
            elif isinstance(v, int):
                logger.info(f"  {k:<20}: {v}")
        logger.info("=" * 50)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    logger = get_logger("ToothMatchNet.predict")

    # Apply CLI overrides to infer config
    ic = CFG.infer
    if args.threshold  is not None: ic.threshold   = args.threshold
    if args.device     is not None: ic.device      = args.device
    if args.batch_size is not None: ic.batch_size  = args.batch_size
    if args.no_amp:                 ic.use_amp     = False
    if args.tta:                    ic.tta_enabled = True

    device = torch.device(
        ic.device if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    model = load_model(args.checkpoint, CFG, device)

    data_dir   = Path(args.data_dir)
    threshold  = ic.threshold
    use_tta    = ic.tta_enabled or args.tta

    if args.sample_id:
        # ---- Single sample mode ----
        result = predict_single(
            model, data_dir, args.sample_id, CFG, device,
            threshold=threshold, use_tta=use_tta
        )
        logger.info("\n" + "=" * 50)
        logger.info(f"  Sample ID  : {result['sample_id']}")
        logger.info(f"  Probability: {result['probability']:.4f}")
        logger.info(f"  Prediction : {'MATCH' if result['prediction'] == 1 else 'NO MATCH'}")
        logger.info("=" * 50)

    else:
        # ---- Batch mode ----
        labels_csv = Path(args.labels_csv) if args.labels_csv else None
        predict_batch(
            model        = model,
            data_dir     = data_dir,
            labels_csv   = labels_csv,
            cfg          = CFG,
            device       = device,
            threshold    = threshold,
            use_tta      = use_tta,
            output_path  = Path(args.output),
        )


if __name__ == "__main__":
    main()
