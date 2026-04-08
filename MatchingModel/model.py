"""
model.py — ToothMatchNet network architecture.

Architecture:
    Input: 4-channel tooth image  +  4-channel eden image
              ↓                           ↓
    Dual-stream ConvNeXt encoder (SEPARATE weights per branch)
              ↓                           ↓
         [B, C, H', W']            [B, C, H', W']
              ↓                           ↓
    Linear projection → tokens     Linear projection → tokens
              ↓
    Bidirectional Cross-Attention Fusion (N layers)
              ↓
    GlobalAvgPool + Concatenate
              ↓
    MLP Head → sigmoid → probability

Key design decisions:
    1. SEPARATE (non-shared) encoders for tooth and eden branches.
       - Tooth images have FIXED orientation (U-shape)
       - Eden images have ARBITRARY orientation (0-360°)
       - Shared weights force the same feature space → wrong inductive bias
       - Separate weights let each branch specialize

    2. 3-channel input (RGB) to leverage ImageNet pretraining fully.
       Depth map is colorized (COLORMAP_JET) in preprocessing, or
       we simply load depth as 3ch by repeating grayscale → RGB.
       This avoids the 4ch stem patching problem which destroys
       pretrained weight alignment.

    3. Two-stream input per branch:
       depth_img  (3ch, depth rendered as RGB) +
       normal_img (3ch, normal map RGB)
       are processed by TWO SEPARATE backbone stems but share upper layers,
       OR more simply: we concatenate depth(repeated to 3ch) + normal(3ch)
       as a 6ch input with a patched stem.

    4. ACTUALLY SIMPLEST robust approach that works with pretrained weights:
       - tooth encoder: takes normal_img (3ch RGB) → ImageNet pretrained ✓
         depth info is encoded as a separate lightweight branch and fused
       - But to keep the architecture clean and the 4ch design intent:
         We use a FRESH random stem (4ch→C) + keep all other pretrained layers.
         The stem is small (4×4 conv), so learning it from scratch is fast.

Dependencies:
    - torch / torchvision  (no timm required)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Supported backbone → (factory_fn, weights_enum, feature_dim)
# ---------------------------------------------------------------------------

_CONVNEXT_REGISTRY = {
    "convnext_tiny":  (tvm.convnext_tiny,  tvm.ConvNeXt_Tiny_Weights.DEFAULT,  768),
    "convnext_small": (tvm.convnext_small, tvm.ConvNeXt_Small_Weights.DEFAULT, 768),
    "convnext_base":  (tvm.convnext_base,  tvm.ConvNeXt_Base_Weights.DEFAULT,  1024),
    "convnext_large": (tvm.convnext_large, tvm.ConvNeXt_Large_Weights.DEFAULT, 1536),
}

_RESNET_REGISTRY = {
    "resnet18":  (tvm.resnet18,  tvm.ResNet18_Weights.DEFAULT,  512),
    "resnet34":  (tvm.resnet34,  tvm.ResNet34_Weights.DEFAULT,  512),
    "resnet50":  (tvm.resnet50,  tvm.ResNet50_Weights.DEFAULT,  2048),
    "resnet101": (tvm.resnet101, tvm.ResNet101_Weights.DEFAULT, 2048),
}

# 合并所有支持的backbone
_BACKBONE_REGISTRY = {**_CONVNEXT_REGISTRY, **_RESNET_REGISTRY}


def _make_convnext_encoder(backbone_name: str,
                            in_channels: int,
                            pretrained: bool) -> Tuple[nn.Module, int]:
    """
    Build a ConvNeXt feature extractor.

    Strategy for in_channels != 3:
        - Load pretrained 3-ch weights
        - Replace stem Conv2d with a new in_channels Conv2d
        - Initialize new stem with KAIMING NORMAL (random, fresh)
          Do NOT copy/tile pretrained weights — that would make channels
          redundant and prevent the model from learning different features
          from depth vs normal channels.
        - Keep all other pretrained weights intact → upper layers benefit
          from ImageNet pretraining even with the new stem.

    Returns:
        encoder : nn.Module  — [B, in_channels, H, W] → [B, feat_dim, H', W']
        feat_dim: int
    """
    if backbone_name not in _CONVNEXT_REGISTRY:
        raise ValueError(
            f"Unknown ConvNeXt backbone '{backbone_name}'. "
            f"Choose from: {list(_CONVNEXT_REGISTRY.keys())}"
        )

    factory_fn, weights_enum, feat_dim = _CONVNEXT_REGISTRY[backbone_name]

    weights = weights_enum if pretrained else None
    full_model = factory_fn(weights=weights)

    # Extract only feature extractor (drop avgpool + classifier)
    features: nn.Sequential = full_model.features

    if in_channels != 3:
        # Patch stem: features[0] is Sequential, features[0][0] is Conv2d(3, C, 4, 4)
        stem_block: nn.Sequential = features[0]
        old_conv: nn.Conv2d = stem_block[0]

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # KEY FIX: Use KAIMING init (random fresh stem)
        # Do NOT tile/repeat pretrained weights — that makes all input channels
        # identical in gradient space and prevents channel specialization.
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        stem_block[0] = new_conv

    return features, feat_dim


def _make_resnet_encoder(backbone_name: str,
                          in_channels: int,
                          pretrained: bool) -> Tuple[nn.Module, int]:
    """
    Build a ResNet feature extractor.

    Strategy for in_channels != 3:
        - Load pretrained 3-ch weights
        - Replace first Conv2d (conv1) with a new in_channels Conv2d
        - Initialize new conv1 with KAIMING NORMAL (random, fresh)
        - Keep all other pretrained weights intact

    Returns:
        encoder : nn.Module  — [B, in_channels, H, W] → [B, feat_dim, H', W']
        feat_dim: int
    """
    if backbone_name not in _RESNET_REGISTRY:
        raise ValueError(
            f"Unknown ResNet backbone '{backbone_name}'. "
            f"Choose from: {list(_RESNET_REGISTRY.keys())}"
        )

    factory_fn, weights_enum, feat_dim = _RESNET_REGISTRY[backbone_name]

    weights = weights_enum if pretrained else None
    full_model = factory_fn(weights=weights)

    # Extract layers before avgpool and fc
    # ResNet structure: conv1, bn1, relu, maxpool, layer1-4
    layers = [
        full_model.conv1,
        full_model.bn1,
        full_model.relu,
        full_model.maxpool,
        full_model.layer1,
        full_model.layer2,
        full_model.layer3,
        full_model.layer4,
    ]
    features = nn.Sequential(*layers)

    if in_channels != 3:
        # Patch conv1: change input channels
        old_conv = full_model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # KAIMING init for new conv1
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        features[0] = new_conv

    return features, feat_dim


def _make_encoder(backbone_name: str,
                   in_channels: int,
                   pretrained: bool) -> Tuple[nn.Module, int]:
    """
    Universal encoder factory supporting both ConvNeXt and ResNet.

    Returns:
        encoder : nn.Module  — [B, in_channels, H, W] → [B, feat_dim, H', W']
        feat_dim: int
    """
    if backbone_name in _CONVNEXT_REGISTRY:
        return _make_convnext_encoder(backbone_name, in_channels, pretrained)
    elif backbone_name in _RESNET_REGISTRY:
        return _make_resnet_encoder(backbone_name, in_channels, pretrained)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choose from: {list(_BACKBONE_REGISTRY.keys())}"
        )


# ---------------------------------------------------------------------------
# Lightweight Adapter: maps 4ch input → 3ch for backbone
# (Alternative to patching stem — keeps ALL pretrained weights intact)
# ---------------------------------------------------------------------------

class InputAdapter(nn.Module):
    """
    Learnable 4ch → 3ch projection placed BEFORE the ConvNeXt backbone.
    This keeps 100% of pretrained weights unchanged in the backbone.

    depth (1ch) + normal (3ch) → adapter → 3ch → backbone

    The adapter is a simple 1×1 conv (fast, minimal params).
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full Encoder Branch: InputAdapter(4ch→3ch) + ConvNeXt(3ch→C)
# ---------------------------------------------------------------------------

class BranchEncoder(nn.Module):
    """
    Complete encoder for one branch (tooth or eden).

    Pipeline: [B, 4, H, W]
        → InputAdapter(4→3)        [B, 3, H, W]
        → Backbone features        [B, feat_dim, H', W']
        → AdaptiveAvgPool2d(1,1)   [B, feat_dim, 1, 1]  (if pool=True)
    
    Supports both ConvNeXt and ResNet backbones.
    """
    def __init__(self, backbone_name: str, pretrained: bool, pool: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.adapter  = InputAdapter(in_channels=4, out_channels=3)

        # Use universal encoder factory
        self.backbone, self.feat_dim = _make_encoder(backbone_name, in_channels=3, pretrained=pretrained)
        self.pool     = nn.AdaptiveAvgPool2d(1) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 4, H, W] → [B, feat_dim, H', W'] or [B, feat_dim] if pool"""
        x = self.adapter(x)        # [B, 3, H, W]
        x = self.backbone(x)       # [B, feat_dim, H', W']
        if self.pool is not None:
            x = self.pool(x).flatten(1)  # [B, feat_dim]
        return x


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    One bidirectional cross-attention block.

    Given query tokens Q (tooth) and key-value tokens KV (eden):
        out_Q  = Q  + MHA(Q,  KV, KV)   # tooth attends to eden
        out_KV = KV + MHA(KV, Q,  Q)    # eden  attends to tooth
    Both streams are then passed through a feed-forward sublayer.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # tooth → eden attention
        self.attn_t2e = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        # eden → tooth attention
        self.attn_e2t = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout,
                                               batch_first=True)

        # FFN for each stream
        self.ffn_tooth = self._make_ffn(embed_dim, dropout)
        self.ffn_eden  = self._make_ffn(embed_dim, dropout)

        self.norm_t1 = nn.LayerNorm(embed_dim)
        self.norm_t2 = nn.LayerNorm(embed_dim)
        self.norm_e1 = nn.LayerNorm(embed_dim)
        self.norm_e2 = nn.LayerNorm(embed_dim)

        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _make_ffn(embed_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self,
                tooth_tokens: torch.Tensor,
                eden_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tooth_tokens: [B, N, D]
            eden_tokens:  [B, N, D]
        Returns:
            tooth_out, eden_out  — both [B, N, D]
        """
        # Tooth attends to Eden
        t_res, _ = self.attn_t2e(tooth_tokens, eden_tokens, eden_tokens)
        tooth_tokens = self.norm_t1(tooth_tokens + self.drop(t_res))
        tooth_tokens = self.norm_t2(tooth_tokens + self.ffn_tooth(tooth_tokens))

        # Eden attends to Tooth
        e_res, _ = self.attn_e2t(eden_tokens, tooth_tokens, tooth_tokens)
        eden_tokens = self.norm_e1(eden_tokens + self.drop(e_res))
        eden_tokens = self.norm_e2(eden_tokens + self.ffn_eden(eden_tokens))

        return tooth_tokens, eden_tokens


# ---------------------------------------------------------------------------
# Cross-Attention Fusion Module
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Projects both branch feature maps to a common token sequence,
    runs N bidirectional cross-attention layers, then aggregates.
    """

    def __init__(self, feat_dim: int, embed_dim: int,
                 num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()

        # Linear projections: backbone feat_dim → embed_dim
        self.proj_tooth = nn.Linear(feat_dim, embed_dim)
        self.proj_eden  = nn.Linear(feat_dim, embed_dim)

        # Stacked cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm_tooth = nn.LayerNorm(embed_dim)
        self.norm_eden  = nn.LayerNorm(embed_dim)

    def forward(self,
                tooth_feat: torch.Tensor,
                eden_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tooth_feat: [B, C, H, W]
            eden_feat:  [B, C, H, W]
        Returns:
            fused: [B, 2 * embed_dim]
        """
        # Flatten spatial dims → token sequence [B, H*W, C]
        tooth_tokens = tooth_feat.flatten(2).transpose(1, 2)   # [B, N, C]
        eden_tokens  = eden_feat.flatten(2).transpose(1, 2)

        # Project to embed_dim
        tooth_tokens = self.proj_tooth(tooth_tokens)            # [B, N, D]
        eden_tokens  = self.proj_eden(eden_tokens)

        # Bidirectional cross-attention
        for block in self.blocks:
            tooth_tokens, eden_tokens = block(tooth_tokens, eden_tokens)

        # Normalise & aggregate (mean over tokens)
        tooth_tokens = self.norm_tooth(tooth_tokens)
        eden_tokens  = self.norm_eden(eden_tokens)

        tooth_vec = tooth_tokens.mean(dim=1)                    # [B, D]
        eden_vec  = eden_tokens.mean(dim=1)

        return torch.cat([tooth_vec, eden_vec], dim=1)          # [B, 2D]


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    MLP classification head.

    Optionally accepts an explicit `scale_feature` tensor [B, scale_dim]
    that encodes the log-ratio of tooth/eden resize scales, giving the model
    a direct size-compatibility signal without relying solely on visual features.

    Architecture:
        [fused_vec | scale_feature]  →  Linear → BN → GELU → Dropout → ...  → logit
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 64),
                 dropout: float = 0.3,
                 scale_dim: int = 1):          # dimension of scale_feature
        super().__init__()
        # First layer receives fused features + scale feature
        total_in = in_dim + scale_dim
        layers = []
        prev = total_in
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))   # single logit
        self.net = nn.Sequential(*layers)
        self.scale_dim = scale_dim

    def forward(self,
                x: torch.Tensor,
                scale_feature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x             : [B, in_dim]   — fused cross-attention feature
            scale_feature : [B, scale_dim] — log(tooth_scale / eden_scale)
                            If None, replaced with zeros (backward compat).
        Returns:
            logits [B]
        """
        if scale_feature is None:
            scale_feature = torch.zeros(x.shape[0], self.scale_dim,
                                        device=x.device, dtype=x.dtype)
        x = torch.cat([x, scale_feature], dim=1)   # [B, in_dim + scale_dim]
        return self.net(x).squeeze(-1)              # [B]


# ---------------------------------------------------------------------------
# ToothMatchNet — Main Model
# ---------------------------------------------------------------------------

class ToothMatchNet(nn.Module):
    """
    Dual-branch binary matching network for dental images.

    Uses SEPARATE (non-shared) encoders for tooth and eden branches because:
    - Tooth has fixed orientation; eden has arbitrary orientation
    - Shared weights force identical feature extraction → wrong inductive bias
    - Separate weights allow branch-specific specialization

    Each encoder uses InputAdapter(4ch→3ch) + ConvNeXt to fully leverage
    ImageNet pretrained weights without modifying backbone architecture.

    Inputs (per batch item):
        tooth_img : [B, 4, H, W]   — depth(1ch) + normal(3ch) of dentition
        eden_img  : [B, 4, H, W]   — depth(1ch) + normal(3ch) of edentulous jaw

    Output:
        logits    : [B]             — raw logit (apply sigmoid for probability)
    """

    def __init__(self,
                 backbone: str         = "convnext_small",
                 pretrained: bool      = True,
                 attn_embed_dim: int   = 256,
                 attn_num_heads: int   = 8,
                 attn_dropout: float   = 0.1,
                 attn_num_layers: int  = 2,
                 head_hidden_dims: Tuple[int, ...] = (256, 64),
                 head_dropout: float   = 0.3,
                 **kwargs):
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone '{backbone}'. "
                             f"Choose from: {list(_BACKBONE_REGISTRY.keys())}")

        _, _, feat_dim = _BACKBONE_REGISTRY[backbone]

        # SEPARATE encoders for each branch (key design decision)
        self.tooth_encoder = BranchEncoder(backbone, pretrained, pool=False)
        self.eden_encoder  = BranchEncoder(backbone, pretrained, pool=False)

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            feat_dim    = feat_dim,
            embed_dim   = attn_embed_dim,
            num_heads   = attn_num_heads,
            num_layers  = attn_num_layers,
            dropout     = attn_dropout,
        )

        # Classification head
        self.head = ClassificationHead(
            in_dim      = attn_embed_dim * 2,
            hidden_dims = head_hidden_dims,
            dropout     = head_dropout,
        )

        # Init fusion + head weights
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for module in [self.fusion, self.head]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    def forward(self,
                tooth_img: torch.Tensor,
                eden_img: torch.Tensor,
                scale_feature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tooth_img     : [B, 4, H, W]
            eden_img      : [B, 4, H, W]
            scale_feature : [B, 1]  log(tooth_scale / eden_scale)
                            Explicit size-compatibility cue.
                            None → head uses zeros (backward compat / TTA).
        Returns:
            logits [B]  (raw; apply sigmoid to get match probability)
        """
        tooth_feat = self.tooth_encoder(tooth_img)          # [B, C, H', W']
        eden_feat  = self.eden_encoder(eden_img)            # [B, C, H', W']

        fused  = self.fusion(tooth_feat, eden_feat)         # [B, 2D]
        logits = self.head(fused, scale_feature)            # [B]
        return logits

    # ------------------------------------------------------------------
    def predict_proba(self,
                      tooth_img: torch.Tensor,
                      eden_img: torch.Tensor,
                      scale_feature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns sigmoid probabilities [B] in [0, 1]."""
        return torch.sigmoid(self.forward(tooth_img, eden_img, scale_feature))

    # ------------------------------------------------------------------
    def predict(self,
                tooth_img: torch.Tensor,
                eden_img: torch.Tensor,
                scale_feature: Optional[torch.Tensor] = None,
                threshold: float = 0.5) -> torch.Tensor:
        """Returns binary predictions [B] ∈ {0, 1}."""
        return (self.predict_proba(tooth_img, eden_img, scale_feature) >= threshold).long()

    # ------------------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg) -> ToothMatchNet:
    """Build ToothMatchNet from a ModelConfig / CFG object."""
    m_cfg = cfg.model if hasattr(cfg, "model") else cfg
    model = ToothMatchNet(
        backbone         = m_cfg.backbone,
        pretrained       = m_cfg.pretrained,
        attn_embed_dim   = m_cfg.attn_embed_dim,
        attn_num_heads   = m_cfg.attn_num_heads,
        attn_dropout     = m_cfg.attn_dropout,
        attn_num_layers  = m_cfg.attn_num_layers,
        head_hidden_dims = m_cfg.head_hidden_dims,
        head_dropout     = m_cfg.head_dropout,
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from config import CFG

    net = build_model(CFG)
    print(f"Backbone         : {CFG.model.backbone}")
    print(f"Total params     : {net.num_parameters:,}")
    print(f"Trainable params : {net.num_trainable_parameters:,}")

    dummy_tooth = torch.randn(2, 4, 224, 224)
    dummy_eden  = torch.randn(2, 4, 224, 224)
    out = net(dummy_tooth, dummy_eden)
    print(f"Output shape     : {out.shape}")          # → torch.Size([2])
    print(f"Sigmoid output   : {torch.sigmoid(out)}")
