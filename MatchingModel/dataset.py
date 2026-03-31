"""
dataset.py — Dataset and DataLoader utilities for ToothMatchNet.

Directory layout expected::

    data_dir/
        sample_0001/
            xxx_eden_depth.png    ← edentulous jaw depth map  (grayscale → 1ch)
            xxx_eden_normal.png   ← edentulous jaw normal map (RGB      → 3ch)
            xxx_tooth_depth.png   ← dentition depth map       (grayscale → 1ch)
            xxx_tooth_normal.png  ← dentition normal map      (RGB      → 3ch)
        sample_0002/
            ...

    Each sample lives in its own sub-directory named by sample_id.
    Inside the sub-directory, each of the four images is identified by its
    SUFFIX (the part before .png that ends in _eden_depth / _eden_normal /
    _tooth_depth / _tooth_normal).  The prefix before those suffixes can be
    anything — the loader scans for the first file whose name ends with the
    expected suffix.

Branch tensors:
    tooth_img : [4, H, W]  — tooth_depth | tooth_normal
    eden_img  : [4, H, W]  — eden_depth  | eden_normal

Key augmentation philosophy
    ● Tooth branch: direction is FIXED (U-shape), so only small rotations +
      horizontal flip are applied.
    ● Eden branch: orientation is ARBITRARY (0-360°), so full ±180° random
      rotation (+ flips) is applied to make the model rotation-invariant.
    ● Depth maps (1-ch) and normal maps (3-ch) within each branch are augmented
      with IDENTICAL spatial transforms so spatial consistency is preserved.
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Helper: load a 4-channel branch tensor from depth + normal paths
# ---------------------------------------------------------------------------

def load_branch(depth_path: Path, normal_path: Path,
                image_size: Tuple[int, int],
                keep_aspect: bool = True) -> Tuple[torch.Tensor, float]:
    """
    Load and concatenate depth (1ch) + normal (3ch) → [4, H, W] float tensor.
    Values are in [0, 1] before normalisation.

    Args:
        depth_path:  Path to depth PNG (grayscale).
        normal_path: Path to normal PNG (RGB).
        image_size:  Target (H, W) after resizing.
        keep_aspect: If True, resize with padding to preserve aspect ratio.
                     This retains the RELATIVE SIZE of the dental arch, which
                     is a key discriminative cue for tooth-eden matching.
                     If False, stretch to fill (loses size information).

    Returns:
        tensor : [4, H, W] float in [0, 1]
        scale  : float — ratio of (original long side / image_size long side).
                 Can be used as an explicit size feature. 1.0 = no change.
    """
    target_h, target_w = image_size

    # ---- Load depth (grayscale) ----
    depth_pil = Image.open(depth_path).convert("L")
    orig_w, orig_h = depth_pil.size   # PIL: (W, H)

    if keep_aspect:
        # Scale so the longer side fits inside target, pad shorter side
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        depth_pil  = depth_pil.resize((new_w, new_h), Image.BILINEAR)
        normal_pil = Image.open(normal_path).convert("RGB")
        normal_pil = normal_pil.resize((new_w, new_h), Image.BILINEAR)

        # Pad to target size with zeros (black = background / empty)
        pad_top  = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        depth_t  = TF.to_tensor(depth_pil)   # [1, new_h, new_w]
        normal_t = TF.to_tensor(normal_pil)  # [3, new_h, new_w]

        depth_t  = F.pad(depth_t,  [pad_left, target_w - new_w - pad_left,
                                     pad_top,  target_h - new_h - pad_top])
        normal_t = F.pad(normal_t, [pad_left, target_w - new_w - pad_left,
                                     pad_top,  target_h - new_h - pad_top])
    else:
        # Stretch (original behaviour) — loses aspect / size information
        scale = min(target_w / orig_w, target_h / orig_h)
        depth_pil  = depth_pil.resize((target_w, target_h), Image.BILINEAR)
        normal_pil = Image.open(normal_path).convert("RGB")
        normal_pil = normal_pil.resize((target_w, target_h), Image.BILINEAR)
        depth_t    = TF.to_tensor(depth_pil)
        normal_t   = TF.to_tensor(normal_pil)

    return torch.cat([depth_t, normal_t], dim=0), scale   # [4, H, W], float


# ---------------------------------------------------------------------------
# Paired spatial augmentation (same transform applied to depth + normal)
# ---------------------------------------------------------------------------

class BranchAugmentor:
    """
    Applies identical spatial transforms to a 4-channel [depth+normal] tensor.
    The depth channel and normal channels receive the same spatial warp.

    For normal maps: rotation of the surface normal vector is approximated by
    rotating the image spatially (which is sufficient for classification tasks).
    """

    def __init__(self,
                 rotate_degrees: float  = 0.0,
                 hflip_prob: float      = 0.0,
                 vflip_prob: float      = 0.0,
                 # Scale jitter: randomly zoom in/out to simulate different arch sizes
                 # Only applied to the branch whose keep_aspect=True padding leaves room.
                 scale_jitter: float    = 0.0,   # e.g. 0.15 → zoom [0.85, 1.15]
                 color_jitter: bool     = False,
                 brightness: float      = 0.0,
                 contrast: float        = 0.0):
        self.rotate_degrees = rotate_degrees
        self.hflip_prob     = hflip_prob
        self.vflip_prob     = vflip_prob
        self.scale_jitter   = scale_jitter
        self.color_jitter   = color_jitter
        self.brightness     = brightness
        self.contrast       = contrast

    def __call__(self, img_4ch: torch.Tensor) -> torch.Tensor:
        """
        img_4ch: [4, H, W] — applies in-place-free spatial + photometric transforms.

        Scale jitter implementation:
            We zoom the image by a random factor s ∈ [1-scale_jitter, 1+scale_jitter].
            - s > 1: crop center → equivalent to a larger arch filling less canvas
            - s < 1: pad with zeros → equivalent to a smaller arch in a larger canvas
            Combined with keep_aspect loading, this makes the model see different
            relative sizes of the dental arch within the padded canvas, forcing it
            to use shape rather than canvas-fill ratio as the primary cue.
        """
        H, W = img_4ch.shape[-2], img_4ch.shape[-1]

        # 1. Scale jitter (before rotation to avoid border artifacts)
        if self.scale_jitter > 0:
            s = random.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter)
            new_h = max(1, int(round(H * s)))
            new_w = max(1, int(round(W * s)))
            # Resize image content
            img_4ch = F.interpolate(img_4ch.unsqueeze(0), size=(new_h, new_w),
                                    mode="bilinear", align_corners=False).squeeze(0)
            if s > 1.0:
                # Zoomed in: center-crop back to H×W
                top  = (new_h - H) // 2
                left = (new_w - W) // 2
                img_4ch = img_4ch[:, top:top + H, left:left + W]
            else:
                # Zoomed out: pad back to H×W with zeros
                pad_t = (H - new_h) // 2
                pad_l = (W - new_w) // 2
                img_4ch = F.pad(img_4ch,
                                [pad_l, W - new_w - pad_l,
                                 pad_t, H - new_h - pad_t])

        # 2. Random rotation (same angle for all channels)
        if self.rotate_degrees > 0:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            img_4ch = TF.rotate(img_4ch, angle, interpolation=TF.InterpolationMode.BILINEAR,
                                fill=0)

        # 3. Horizontal flip
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            img_4ch = TF.hflip(img_4ch)

        # 4. Vertical flip
        if self.vflip_prob > 0 and random.random() < self.vflip_prob:
            img_4ch = TF.vflip(img_4ch)

        # 5. Color jitter applied only to depth ch + normal chs independently
        #    (jitter is per-channel brightness/contrast only — no hue/saturation)
        if self.color_jitter:
            # depth channel [0:1]
            depth  = img_4ch[0:1]
            normal = img_4ch[1:4]

            if self.brightness > 0:
                b_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
                depth    = torch.clamp(depth  * b_factor, 0, 1)
                normal   = torch.clamp(normal * b_factor, 0, 1)

            if self.contrast > 0:
                c_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                depth    = torch.clamp((depth  - 0.5) * c_factor + 0.5, 0, 1)
                normal   = torch.clamp((normal - 0.5) * c_factor + 0.5, 0, 1)

            img_4ch = torch.cat([depth, normal], dim=0)

        return img_4ch


# ---------------------------------------------------------------------------
# Normalisation transform
# ---------------------------------------------------------------------------

class Normalize4ch(torch.nn.Module):
    """Normalise a 4-channel tensor with per-channel mean/std."""

    def __init__(self, mean: Tuple[float, ...] = (0.5,) * 4,
                 std:  Tuple[float, ...] = (0.5,) * 4):
        super().__init__()
        self.mean = torch.tensor(mean).view(4, 1, 1)
        self.std  = torch.tensor(std).view(4, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ToothMatchDataset(Dataset):
    """
    Dataset for tooth-eden matching.

    Directory layout expected::

        data_dir/
            sample_0001/
                xxx_eden_depth.png
                xxx_eden_normal.png
                xxx_tooth_depth.png
                xxx_tooth_normal.png
            sample_0002/
                ...

    Each sample lives in its own sub-directory named by sample_id.
    Within each sub-directory, files are located by scanning for the first
    PNG whose filename ends with the expected suffix (prefix is arbitrary).

    labels_csv format (header required)::

        sample_id,split,label
        sample_0001,train,1
        sample_0002,train,0
        sample_0045,val,1
        sample_0046,test,0

        Columns:
            sample_id : str  — must match the sub-directory name exactly
            split     : str  — one of  train / val / test
            label     : int  — 0 (no match)  or  1 (match)
    """

    # Suffixes used to identify each of the four image roles.
    SUFFIXES = {
        "eden_depth":   "_eden_depth.png",
        "eden_normal":  "_eden_normal.png",
        "tooth_depth":  "_tooth_depth.png",
        "tooth_normal": "_tooth_normal.png",
    }

    def __init__(self,
                 data_dir:   Path,
                 labels_csv: Path,
                 image_size: Tuple[int, int]     = (224, 224),
                 tooth_augmentor: Optional[BranchAugmentor] = None,
                 eden_augmentor:  Optional[BranchAugmentor] = None,
                 normalizer: Optional[Normalize4ch]         = None,
                 split: str = "train"):
        """
        Args:
            data_dir:         Root directory that contains one sub-directory per sample.
            labels_csv:       Path to labels.csv with columns: sample_id, split, label.
                              Only rows whose `split` value matches this dataset's split
                              will be loaded — so one shared CSV covers all three splits.
            image_size:       (H, W) to resize images to.
            tooth_augmentor:  BranchAugmentor for the tooth branch (or None).
            eden_augmentor:   BranchAugmentor for the eden branch (or None).
            normalizer:       Normalize4ch instance (or None).
            split:            Which split to load: "train" / "val" / "test".
        """
        self.data_dir        = Path(data_dir)
        self.image_size      = image_size
        self.tooth_augmentor = tooth_augmentor
        self.eden_augmentor  = eden_augmentor
        self.normalizer      = normalizer
        self.split           = split

        self.samples = self._load_samples(labels_csv)

    # ------------------------------------------------------------------
    @staticmethod
    def _find_file_by_suffix(sample_dir: Path, suffix: str) -> Optional[Path]:
        """
        Return the unique file inside sample_dir whose filename ends with `suffix`.

        - The prefix before the suffix can be anything (e.g. 'scanA', 'nA', 'sc').
        - Matching is done purely on the suffix string (case-sensitive).
        - If MULTIPLE files match the same suffix, a warning is printed and the
          lexicographically first one is used (deterministic across runs).
        - Returns None if no matching file is found.
        """
        matches = sorted([
            p for p in sample_dir.iterdir()
            if p.is_file() and p.name.endswith(suffix)
        ])
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            print(
                f"[Dataset] WARNING: multiple files match suffix '{suffix}' "
                f"in {sample_dir}:\n"
                + "\n".join(f"    {m.name}" for m in matches)
                + f"\n  → using '{matches[0].name}' (lexicographic first)"
            )
        return matches[0]

    # ------------------------------------------------------------------
    def _load_samples(self, labels_csv: Path) -> List[Dict]:
        """
        Parse labels_csv and return only the rows that belong to self.split.

        CSV format (header required):
            sample_id,split,label
            sample_0001,train,1
            sample_0002,val,0
            ...

        Rules:
          - Rows whose `split` column != self.split are silently skipped.
          - Sub-directories that don't exist under self.data_dir produce a WARNING.
          - Samples with any missing image file produce a WARNING and are skipped.
        """
        samples = []
        has_split_col = None   # determined from the first valid row

        with open(labels_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip completely empty rows (e.g. trailing newline in CSV)
                if all(v is None or (v and v.strip() == "") for v in row.values()):
                    continue

                # Normalise key names: strip BOM / whitespace from all keys
                row = {(k.strip().lstrip("\ufeff") if k else k): v
                       for k, v in row.items()}

                # Validate required columns on first real row
                if "sample_id" not in row:
                    raise KeyError(
                        f"'sample_id' column not found in {labels_csv}. "
                        f"Detected columns: {list(row.keys())}. "
                        "Expected header: sample_id,split,label"
                    )
                if "label" not in row:
                    raise KeyError(
                        f"'label' column not found in {labels_csv}. "
                        f"Detected columns: {list(row.keys())}. "
                        "Expected header: sample_id,split,label"
                    )

                # Detect whether 'split' column exists
                if has_split_col is None:
                    has_split_col = "split" in row

                # ---- Filter by split ----
                if has_split_col:
                    row_split = row.get("split", "").strip()
                    if row_split != self.split:
                        continue   # belongs to a different split — silent skip
                # (if no split column, load all rows — backward compat)

                sid   = row["sample_id"].strip()
                label = int(row["label"].strip())

                # Each sample lives in its own sub-directory inside data_dir
                sample_dir = self.data_dir / sid
                if not sample_dir.is_dir():
                    print(f"[Dataset] WARNING: sub-directory not found for "
                          f"'{sid}' in {self.data_dir}")
                    continue

                # Locate each of the four images by suffix (prefix is arbitrary)
                paths = {}
                missing = []
                for key, sfx in self.SUFFIXES.items():
                    found = self._find_file_by_suffix(sample_dir, sfx)
                    if found is None:
                        missing.append(f"*{sfx}")
                    else:
                        paths[key] = found

                if missing:
                    print(f"[Dataset] WARNING: missing image(s) for '{sid}' "
                          f"in {sample_dir}: {missing} — sample skipped.")
                    continue

                samples.append({"sample_id": sid, "label": label, **paths})

        if not samples:
            raise RuntimeError(
                f"No valid samples found for split='{self.split}' in {self.data_dir}.\n"
                f"  labels_csv : {labels_csv}\n"
                f"  Tip: make sure labels.csv has a 'split' column with values "
                f"train/val/test, and the corresponding sub-directories exist."
            )
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # Load raw 4-channel branch tensors [4, H, W] ∈ [0, 1]
        # keep_aspect=True → pads to preserve size information in the canvas fill ratio
        tooth_img, tooth_scale = load_branch(
            s["tooth_depth"], s["tooth_normal"], self.image_size, keep_aspect=True)
        eden_img, eden_scale = load_branch(
            s["eden_depth"],  s["eden_normal"],  self.image_size, keep_aspect=True)

        # Apply augmentations (includes optional scale jitter)
        if self.tooth_augmentor is not None:
            tooth_img = self.tooth_augmentor(tooth_img)
        if self.eden_augmentor is not None:
            eden_img  = self.eden_augmentor(eden_img)

        # Normalise
        if self.normalizer is not None:
            tooth_img = self.normalizer(tooth_img)
            eden_img  = self.normalizer(eden_img)

        label = torch.tensor(s["label"], dtype=torch.float32)

        # Scale features: log-ratio of (tooth_scale / eden_scale).
        # If tooth and eden are the same real-world size → ratio ≈ 1 → log ≈ 0.
        # The model can use this as an explicit size-compatibility cue.
        # Both scales are the resize factor (original → canvas), so:
        #   large scale = small original object (fits easily)
        #   small scale = large original object (tight fit)
        # log(tooth_scale / eden_scale) > 0 → tooth is smaller than eden
        # log(tooth_scale / eden_scale) < 0 → tooth is larger than eden
        log_scale_ratio = torch.tensor(
            [float(np.log(tooth_scale / max(eden_scale, 1e-6)))],
            dtype=torch.float32)

        return {
            "tooth_img":       tooth_img,        # [4, H, W]
            "eden_img":        eden_img,          # [4, H, W]
            "scale_feature":   log_scale_ratio,  # [1]  explicit size-compatibility cue
            "label":           label,             # scalar float
            "sample_id":       s["sample_id"],
        }

    # ------------------------------------------------------------------
    @property
    def labels(self) -> List[int]:
        return [s["label"] for s in self.samples]

    @property
    def pos_weight(self) -> float:
        """Positive class weight for BCEWithLogitsLoss (# neg / # pos)."""
        labels = self.labels
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        return n_neg / max(n_pos, 1)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_augmentors(cfg) -> Tuple[BranchAugmentor, BranchAugmentor,
                                    BranchAugmentor, BranchAugmentor]:
    """
    Returns (tooth_train_aug, eden_train_aug, tooth_val_aug, eden_val_aug).
    Val augmentors are identity (no-ops).
    """
    dc = cfg.data if hasattr(cfg, "data") else cfg

    tooth_train = BranchAugmentor(
        rotate_degrees = dc.tooth_rotate_degrees,
        hflip_prob     = dc.tooth_hflip_prob,
        vflip_prob     = dc.tooth_vflip_prob,
        scale_jitter   = dc.tooth_scale_jitter,
        color_jitter   = dc.tooth_color_jitter,
        brightness     = dc.tooth_color_jitter_brightness,
        contrast       = dc.tooth_color_jitter_contrast,
    )
    eden_train = BranchAugmentor(
        rotate_degrees = dc.eden_rotate_degrees,
        hflip_prob     = dc.eden_hflip_prob,
        vflip_prob     = dc.eden_vflip_prob,
        scale_jitter   = dc.eden_scale_jitter,
        color_jitter   = dc.eden_color_jitter,
        brightness     = dc.eden_color_jitter_brightness,
        contrast       = dc.eden_color_jitter_contrast,
    )
    # Val/test: no spatial augmentation (identity augmentors)
    tooth_val = BranchAugmentor()
    eden_val  = BranchAugmentor()

    return tooth_train, eden_train, tooth_val, eden_val


def build_normalizer(cfg) -> Normalize4ch:
    dc = cfg.data if hasattr(cfg, "data") else cfg
    return Normalize4ch(mean=dc.normalize_mean, std=dc.normalize_std)


def build_weighted_sampler(dataset: ToothMatchDataset) -> WeightedRandomSampler:
    labels  = dataset.labels
    n_pos   = sum(labels)
    n_neg   = len(labels) - n_pos
    w_pos   = 1.0 / max(n_pos, 1)
    w_neg   = 1.0 / max(n_neg, 1)
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    """
    dc = cfg.data  if hasattr(cfg, "data")  else cfg
    tc = cfg.train if hasattr(cfg, "train") else cfg

    tooth_train_aug, eden_train_aug, tooth_val_aug, eden_val_aug = build_augmentors(cfg)
    normalizer = build_normalizer(cfg)

    from config import TRAIN_DIR, VAL_DIR, TEST_DIR, LABELS_CSV  # noqa

    train_ds = ToothMatchDataset(
        TRAIN_DIR, LABELS_CSV,
        image_size      = dc.image_size,
        tooth_augmentor = tooth_train_aug,
        eden_augmentor  = eden_train_aug,
        normalizer      = normalizer,
        split           = "train",
    )
    val_ds = ToothMatchDataset(
        VAL_DIR, LABELS_CSV,
        image_size      = dc.image_size,
        tooth_augmentor = tooth_val_aug,
        eden_augmentor  = eden_val_aug,
        normalizer      = normalizer,
        split           = "val",
    )
    test_ds = ToothMatchDataset(
        TEST_DIR, LABELS_CSV,
        image_size      = dc.image_size,
        tooth_augmentor = tooth_val_aug,
        eden_augmentor  = eden_val_aug,
        normalizer      = normalizer,
        split           = "test",
    )

    sampler = build_weighted_sampler(train_ds) if tc.use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size      = tc.batch_size,
        sampler         = sampler,
        shuffle         = sampler is None,
        num_workers     = dc.num_workers,
        pin_memory      = dc.pin_memory,
        prefetch_factor = dc.prefetch_factor if dc.num_workers > 0 else None,
        drop_last       = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size      = tc.batch_size,
        shuffle         = False,
        num_workers     = dc.num_workers,
        pin_memory      = dc.pin_memory,
        prefetch_factor = dc.prefetch_factor if dc.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size      = tc.batch_size,
        shuffle         = False,
        num_workers     = dc.num_workers,
        pin_memory      = dc.pin_memory,
        prefetch_factor = dc.prefetch_factor if dc.num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
