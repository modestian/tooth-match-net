# ToothMatchNet

**Binary matching network** that determines whether a given dental prosthesis (**dentition / tooth arch**) fits an **edentulous jaw** from depth + normal map images.

---

## Project Structure

```
ToothMatchNet/
├── MatchingData/
│   ├── train/                  # Training sample images
│   ├── val/                    # Validation sample images
│   ├── test/                   # Test sample images
│   └── labels.csv              # sample_id, label (0=no-match, 1=match)
├── MatchingCheckpoints/        # Saved model checkpoints (auto-created)
├── MatchingModel/
│   ├── config.py               # All hyper-parameters (model / data / train / infer)
│   ├── model.py                # Network architecture
│   ├── dataset.py              # Dataset & DataLoader
│   ├── losses.py               # Loss functions
│   ├── train.py                # Training script
│   ├── predict.py              # Inference script
│   └── utils.py                # Utilities (metrics, checkpoints, logging…)
├── requirements.txt
└── README.md
```

---

## Architecture Overview

```
Input per branch: 4-channel image [B, 4, H, W]
  = depth map (1ch, grayscale) + normal map (3ch, RGB)

Two branches (shared ConvNeXt-Small encoder):
  tooth branch → [B, C, H', W']
  eden  branch → [B, C, H', W']

Cross-Attention Fusion (bidirectional, 2 layers, 8 heads, dim=256):
  tooth tokens attend to eden tokens  AND  eden tokens attend to tooth tokens
  → mean-pool each stream → concatenate → [B, 512]

Classification Head (MLP):
  512 → 128 → 1 → sigmoid → match probability ∈ [0, 1]
```

### Key Design Choices

| Challenge | Solution |
|-----------|----------|
| Eden jaw orientation is **arbitrary** (0–360°) | ±180° random rotation augmentation on eden branch during training |
| Tooth arch direction is **fixed** (U-shape) | Only ±10° jitter + horizontal flip on tooth branch |
| **Test-time** arbitrary orientation | TTA: average predictions over 0°/90°/180°/270° rotations + hflip |
| Class imbalance | `WeightedRandomSampler` + `BCEFocalLoss` + positive class weight |
| 4-channel input ≠ ImageNet 3-channel | Stem conv replaced; pretrained RGB weights averaged & tiled to 4ch |

---

## Image File & Directory Convention

Each `sample_id` corresponds to a **sub-directory** under its split folder.  
Inside that sub-directory, the four images are identified by their **suffix**;
the prefix before the suffix can be anything (e.g. a scan ID, timestamp, etc.).

```
MatchingData/
├── train/
│   ├── sample_0001/
│   │   ├── xxx_eden_depth.png    ← edentulous jaw depth map  (grayscale)
│   │   ├── xxx_eden_normal.png   ← edentulous jaw normal map (RGB)
│   │   ├── xxx_tooth_depth.png   ← dentition depth map       (grayscale)
│   │   └── xxx_tooth_normal.png  ← dentition normal map      (RGB)
│   ├── sample_0002/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── labels.csv
```

Required filename suffixes (the loader searches for a file ending with each):

| Suffix | Role |
|--------|------|
| `_eden_depth.png`  | Edentulous jaw depth map (grayscale, 1-channel) |
| `_eden_normal.png` | Edentulous jaw normal map (RGB, 3-channel) |
| `_tooth_depth.png` | Dentition depth map (grayscale, 1-channel) |
| `_tooth_normal.png`| Dentition normal map (RGB, 3-channel) |

---

## labels.csv Format

```csv
sample_id,label
sample_001,1
sample_002,1
sample_101,0
sample_102,0
```

- **1** → the tooth arch **matches** the edentulous jaw  
- **0** → the tooth arch does **not** match

---

## Quick Start

### 1. Install dependencies

```bash
# Install PyTorch with CUDA 12.8 first (visit https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Then install the rest
pip install -r requirements.txt
```

### 2. Prepare data

Create one sub-directory per sample inside the split folder, then place the
four images inside it.  Ensure `labels.csv` covers every `sample_id`.

```
MatchingData/
├── train/
│   ├── sample_0001/
│   │   ├── scanA_eden_depth.png
│   │   ├── scanA_eden_normal.png
│   │   ├── scanA_tooth_depth.png
│   │   └── scanA_tooth_normal.png
│   ├── sample_0002/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── labels.csv
```

### 3. Train

```bash
cd ToothMatchNet/MatchingModel

# Default training (ConvNeXt-Small, 100 epochs, lr=1e-4)
python train.py

# Custom settings
python train.py --epochs 50 --lr 3e-4 --backbone convnext_base --batch-size 16

# Resume from best checkpoint
python train.py --resume

# Disable AMP (for debugging)
python train.py --no-amp
```

### 4. Predict

```bash
cd ToothMatchNet/MatchingModel

# Single sample
python predict.py \
    --sample-id sample_001 \
    --data-dir ../MatchingData/test

# Batch prediction with evaluation + TTA
python predict.py \
    --data-dir ../MatchingData/test \
    --labels-csv ../MatchingData/labels.csv \
    --output ../results.csv \
    --tta

# Specify checkpoint
python predict.py \
    --checkpoint ../MatchingCheckpoints/best.pth \
    --data-dir ../MatchingData/test \
    --output ../results.csv
```

---

## Configuration

All parameters are in `MatchingModel/config.py` under the `CFG` singleton:

```python
# Model
CFG.model.backbone        = "convnext_small"   # or convnext_tiny / convnext_base
CFG.model.attn_embed_dim  = 256
CFG.model.attn_num_heads  = 8
CFG.model.attn_num_layers = 2

# Training
CFG.train.epochs          = 100
CFG.train.batch_size      = 32
CFG.train.learning_rate   = 1e-4
CFG.train.backbone_lr_scale = 0.1   # backbone trained at lr * 0.1
CFG.train.loss_type       = "bce_focal"
CFG.train.use_amp         = True    # automatic mixed precision
CFG.train.use_weighted_sampler = True

# Augmentation
CFG.data.eden_rotate_degrees  = 180.0   # full rotation for eden branch
CFG.data.tooth_rotate_degrees = 10.0   # small jitter for tooth branch

# Inference / TTA
CFG.infer.tta_enabled   = True
CFG.infer.tta_rotations = (0, 90, 180, 270)
CFG.infer.threshold     = 0.5
```

---

## GPU Memory Requirements

| Backbone       | Image Size | Batch Size | VRAM (est.) |
|----------------|-----------|------------|-------------|
| convnext_tiny  | 224×224   | 32         | ~6 GB       |
| convnext_small | 224×224   | 32         | ~9 GB       |
| convnext_base  | 224×224   | 32         | ~14 GB      |
| convnext_small | 224×224   | 64         | ~16 GB      |
| convnext_base  | 224×224   | 64         | ~22 GB      |

All variants comfortably fit on RTX 4090 (24 GB) and RTX 5090 (32 GB).

---

## TensorBoard

```bash
tensorboard --logdir ToothMatchNet/MatchingCheckpoints/tb_logs
```
