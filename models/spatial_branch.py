"""
models/spatial_branch.py — EfficientNet-B0 spatial branch
Processes RGB face crops → feature vector for fusion
"""

import torch
import torch.nn as nn
import timm


class SpatialBranch(nn.Module):
    """
    EfficientNet-B0 backbone pretrained on ImageNet.
    Input  : (B, 3, 224, 224) normalized RGB face crop
    Output : (B, feature_dim) feature vector  [extract_features=True]
             (B, 2)           logits           [extract_features=False]
    """

    def __init__(self,
                 pretrained:       bool = True,
                 feature_dim:      int  = 256,
                 dropout:          float = 0.3,
                 extract_features: bool = True):
        """
        Args:
            pretrained       : load ImageNet weights
            feature_dim      : size of output feature vector
            dropout          : dropout rate before classifier head
            extract_features : True  → return feature vector (for fusion)
                               False → return logits (standalone classifier)
        """
        super().__init__()
        self.extract_features = extract_features
        self.feature_dim      = feature_dim

        # ── Backbone ──────────────────────────────────────────────────────
        # num_classes=0 removes the default classifier head
        # → output is the raw pooled feature vector (1280-d for B0)
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,          # remove head → raw features
            global_pool="avg",      # global average pooling
        )
        backbone_dim = self.backbone.num_features   # 1280 for EfficientNet-B0

        # ── Feature projection head ───────────────────────────────────────
        # Projects 1280-d backbone output → feature_dim
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Standalone classification head ────────────────────────────────
        # Only used when extract_features=False (spatial-only ablation)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 224, 224)
        returns:
            features : (B, feature_dim)   if extract_features=True
            logits   : (B, 2)             if extract_features=False
        """
        x = self.backbone(x)          # (B, 1280)
        x = self.feature_head(x)      # (B, feature_dim)

        if self.extract_features:
            return x                  # (B, feature_dim) → goes to fusion
        return self.classifier(x)     # (B, 2)           → standalone mode

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze EfficientNet backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        status = "frozen" if freeze else "unfrozen"
        print(f"  [SpatialBranch] Backbone {status}")

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Test 1: feature extraction mode (used in dual-branch) ─────────────
    print("Test 1 — Feature extraction mode")
    model = SpatialBranch(pretrained=True,
                          feature_dim=256,
                          extract_features=True).to(device)
    model.eval()

    x   = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)

    print(f"  Input  : {x.shape}")
    print(f"  Output : {out.shape}   ← feature vector for fusion")
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"

    params = model.count_parameters()
    print(f"  Params : {params['total']:,} total | "
          f"{params['trainable']:,} trainable")

    # ── Test 2: standalone classifier mode (used in ablation) ─────────────
    print("\nTest 2 — Standalone classifier mode")
    model2 = SpatialBranch(pretrained=True,
                           feature_dim=256,
                           extract_features=False).to(device)
    model2.eval()

    with torch.no_grad():
        logits = model2(x)

    print(f"  Input  : {x.shape}")
    print(f"  Output : {logits.shape}   ← logits (real vs fake)")
    assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"

    # ── Test 3: backbone freezing ──────────────────────────────────────────
    print("\nTest 3 — Backbone freezing")
    model.freeze_backbone(True)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Frozen params : {frozen:,}")

    model.freeze_backbone(False)
    unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable after unfreeze: {unfrozen:,}")

    # ── Test 4: gradient flow check ───────────────────────────────────────
    print("\nTest 4 — Gradient flow")
    model3 = SpatialBranch(pretrained=False,
                           extract_features=False).to(device)
    model3.train()
    x_grad  = torch.randn(4, 3, 224, 224).to(device)
    labels  = torch.randint(0, 2, (4,)).to(device)
    logits  = model3(x_grad)
    loss    = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    print(f"  Loss   : {loss.item():.4f}")
    print(f"  Grads  : OK")

    print("\n  spatial_branch.py — all tests passed!")
