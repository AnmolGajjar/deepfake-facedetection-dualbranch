"""
models/dual_branch.py — Dual-Branch Fusion Model
Combines SpatialBranch (RGB) + FrequencyBranch (FFT) via an Attention Gate

Architecture overview
─────────────────────
                 ┌─────────────────────┐
  RGB (B,3,H,W)  │   SpatialBranch     │ → (B, D)  spatial_feat
                 │   EfficientNet-B0   │
                 └─────────────────────┘
                           │
                           ▼
                 ┌─────────────────────┐     ┌──────────────────┐
                 │   AttentionGate     │────▶│  Weighted sum    │→(B, D)
                 └─────────────────────┘     │  α·s + β·f       │
                           ▲                 └──────────────────┘
                 ┌─────────────────────┐              │
  FFT (B,1,H,W)  │   FrequencyBranch   │ → (B, D)     ▼
                 │   Lightweight CNN   │     ┌──────────────────┐
                 └─────────────────────┘     │   Classifier     │→(B,2)
                                             └──────────────────┘

Attention Gate
──────────────
The gate receives the *concatenation* of both feature vectors (B, 2D)
and produces two scalar attention weights [α, β] via a softmax.
The fused representation is then:

    fused = α · spatial_feat + β · freq_feat   (B, D)

This is fully differentiable — the model learns per-sample which branch
to trust more depending on the type of manipulation present.

Fusion modes (controlled by config.model.fusion)
──────────────────────────────────────────────────
  "attention_gate"  — weighted combination via learned gate  (default)
  "concat"          — concatenate both features, project to D

Ablation modes (for Step 7)
────────────────────────────
  branch="both"     — full dual-branch model
  branch="spatial"  — spatial branch only (freq branch is bypassed)
  branch="freq"     — freq branch only (spatial branch is bypassed)

When a branch is bypassed its parameters still exist (so weights can be
loaded consistently), but its output is replaced with zeros so the
attention gate is forced to rely entirely on the active branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .spatial_branch import SpatialBranch
from .freq_branch import FrequencyBranch


# ─────────────────────────────────────────────
# Attention Gate
# ─────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Learns per-sample attention weights over two feature streams.

    Input  : concat of (spatial_feat, freq_feat)  → (B, 2 * feature_dim)
    Output : attention weights α, β  summing to 1  → (B, 2)

    The gate is a 2-layer MLP with a bottleneck:
        Linear(2D → D//2) → ReLU → Linear(D//2 → 2) → Softmax
    """

    def __init__(self, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        bottleneck = max(feature_dim // 4, 32)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 2),
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused   : (B, feature_dim) — weighted combination
            weights : (B, 2)           — [α, β] attention weights (for logging)
        """
        combined = torch.cat([spatial_feat, freq_feat], dim=1)   # (B, 2D)
        weights  = F.softmax(self.gate(combined), dim=1)          # (B, 2)
        alpha    = weights[:, 0:1]     # (B, 1) — spatial weight
        beta     = weights[:, 1:2]     # (B, 1) — freq weight
        fused    = alpha * spatial_feat + beta * freq_feat        # (B, D)
        return fused, weights


# ─────────────────────────────────────────────
# Concat Fusion (simpler baseline)
# ─────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """
    Concatenate both feature vectors and project back to feature_dim.
    Simpler than attention gate — useful as a fusion baseline.
    """

    def __init__(self, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused   : (B, feature_dim)
            weights : (B, 2) — uniform [0.5, 0.5] placeholder for API compat
        """
        combined = torch.cat([spatial_feat, freq_feat], dim=1)   # (B, 2D)
        fused    = self.proj(combined)                            # (B, D)
        # Return uniform weights so logging code works identically
        weights  = torch.full((fused.size(0), 2), 0.5,
                              device=fused.device, dtype=fused.dtype)
        return fused, weights


# ─────────────────────────────────────────────
# Dual-Branch Fusion Model
# ─────────────────────────────────────────────

class DualBranchModel(nn.Module):
    """
    Full dual-branch deepfake detector.

    Args:
        feature_dim      : shared feature dimension for both branches (default 256)
        freq_channels    : channel widths for FrequencyBranch conv stack
        dropout          : dropout rate used in both branches and fusion
        fusion           : "attention_gate" | "concat"
        spatial_pretrained: load ImageNet weights for EfficientNet-B0
        branch           : "both" | "spatial" | "freq"
                           Controls which branches are active.
                           In "spatial" or "freq" mode the other branch's
                           output is zeroed — useful for ablation studies.
    """

    def __init__(
        self,
        feature_dim:        int   = 256,
        freq_channels:      list  = None,
        dropout:            float = 0.3,
        fusion:             str   = "attention_gate",
        spatial_pretrained: bool  = True,
        branch:             str   = "both",
    ):
        super().__init__()
        if freq_channels is None:
            freq_channels = [32, 64, 128]

        assert branch in ("both", "spatial", "freq"), \
            f"branch must be 'both', 'spatial', or 'freq', got '{branch}'"
        assert fusion in ("attention_gate", "concat"), \
            f"fusion must be 'attention_gate' or 'concat', got '{fusion}'"

        self.feature_dim = feature_dim
        self.branch      = branch
        self.fusion_type = fusion

        # ── Branches ──────────────────────────────────────────────────────
        self.spatial_branch = SpatialBranch(
            pretrained=spatial_pretrained,
            feature_dim=feature_dim,
            dropout=dropout,
            extract_features=True,          # always return feature vector
        )
        self.freq_branch = FrequencyBranch(
            freq_channels=freq_channels,
            feature_dim=feature_dim,
            dropout=dropout,
            extract_features=True,          # always return feature vector
        )

        # ── Fusion module ──────────────────────────────────────────────────
        if fusion == "attention_gate":
            self.fusion = AttentionGate(feature_dim=feature_dim, dropout=dropout)
        else:
            self.fusion = ConcatFusion(feature_dim=feature_dim, dropout=dropout)

        # ── Final classifier ───────────────────────────────────────────────
        # A two-layer head with dropout for regularisation
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2),
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        spatial: torch.Tensor,
        freq:    torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            spatial : (B, 3, H, W) — normalized RGB face crop
            freq    : (B, 1, H, W) — log-magnitude FFT spectrum

        Returns dict with keys:
            "logits"          : (B, 2)  — raw class scores
            "spatial_feat"    : (B, D)  — spatial branch features
            "freq_feat"       : (B, D)  — frequency branch features
            "fused_feat"      : (B, D)  — post-fusion features
            "attention_weights": (B, 2) — [α_spatial, β_freq]
        """
        # ── Extract features from each branch ──────────────────────────
        spatial_feat = self.spatial_branch(spatial)   # (B, D)
        freq_feat    = self.freq_branch(freq)          # (B, D)

        # ── Ablation: zero-out the inactive branch ──────────────────────
        if self.branch == "spatial":
            freq_feat = torch.zeros_like(freq_feat)
        elif self.branch == "freq":
            spatial_feat = torch.zeros_like(spatial_feat)

        # ── Fuse ────────────────────────────────────────────────────────
        fused_feat, attn_weights = self.fusion(spatial_feat, freq_feat)

        # ── Classify ────────────────────────────────────────────────────
        logits = self.classifier(fused_feat)          # (B, 2)

        return {
            "logits":            logits,
            "spatial_feat":      spatial_feat,
            "freq_feat":         freq_feat,
            "fused_feat":        fused_feat,
            "attention_weights": attn_weights,
        }

    # ── Utilities ─────────────────────────────────────────────────────────

    def freeze_spatial_backbone(self, freeze: bool = True):
        """Freeze / unfreeze EfficientNet backbone only (not the head)."""
        self.spatial_branch.freeze_backbone(freeze)

    def set_branch_mode(self, branch: str):
        """Switch ablation mode at inference time without reloading weights."""
        assert branch in ("both", "spatial", "freq")
        self.branch = branch
        print(f"  [DualBranchModel] branch mode → '{branch}'")

    def count_parameters(self) -> dict:
        def _count(module):
            total     = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters()
                            if p.requires_grad)
            return {"total": total, "trainable": trainable}

        return {
            "spatial_branch": _count(self.spatial_branch),
            "freq_branch":    _count(self.freq_branch),
            "fusion":         _count(self.fusion),
            "classifier":     _count(self.classifier),
            "total":          _count(self),
        }


# ─────────────────────────────────────────────
# Factory — build from config dict
# ─────────────────────────────────────────────

def build_model(cfg: dict, branch: str = "both") -> DualBranchModel:
    """
    Instantiate DualBranchModel from a config dict (loaded from config.yaml).

    Args:
        cfg    : full config dict
        branch : "both" | "spatial" | "freq" — for ablation studies
    """
    m = cfg["model"]
    return DualBranchModel(
        feature_dim=        m.get("feature_dim",        256),
        freq_channels=      m.get("freq_channels",      [32, 64, 128]),
        dropout=            m.get("dropout",             0.3),
        fusion=             m.get("fusion",              "attention_gate"),
        spatial_pretrained= m.get("spatial_pretrained", True),
        branch=             branch,
    )


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B = 4   # batch size for all tests

    # Dummy inputs
    spatial = torch.randn(B, 3, 224, 224).to(device)
    freq    = torch.randn(B, 1, 224, 224).to(device)

    # ── Test 1: Attention gate fusion (full dual-branch) ──────────────────
    print("Test 1 — Attention gate fusion (both branches)")
    model = DualBranchModel(
        feature_dim=256,
        freq_channels=[32, 64, 128],
        dropout=0.3,
        fusion="attention_gate",
        spatial_pretrained=True,
        branch="both",
    ).to(device)
    model.eval()

    with torch.no_grad():
        out = model(spatial, freq)

    print(f"  logits           : {out['logits'].shape}")
    print(f"  spatial_feat     : {out['spatial_feat'].shape}")
    print(f"  freq_feat        : {out['freq_feat'].shape}")
    print(f"  fused_feat       : {out['fused_feat'].shape}")
    print(f"  attention_weights: {out['attention_weights'].shape}")
    print(f"  attn sample      : {out['attention_weights'][0].tolist()}")

    assert out["logits"].shape            == (B, 2)
    assert out["spatial_feat"].shape      == (B, 256)
    assert out["freq_feat"].shape         == (B, 256)
    assert out["fused_feat"].shape        == (B, 256)
    assert out["attention_weights"].shape == (B, 2)
    # Attention weights must sum to 1
    assert torch.allclose(out["attention_weights"].sum(dim=1),
                          torch.ones(B, device=device), atol=1e-5)

    # Parameter breakdown
    params = model.count_parameters()
    print(f"\n  Parameter breakdown:")
    for k, v in params.items():
        if isinstance(v, dict):
            print(f"    {k:<20}: {v['total']:>10,} total  "
                  f"| {v['trainable']:>10,} trainable")
        else:
            pass
    print(f"    {'TOTAL':<20}: {params['total']['total']:>10,} total  "
          f"| {params['total']['trainable']:>10,} trainable")

    # ── Test 2: Concat fusion ─────────────────────────────────────────────
    print("\nTest 2 — Concat fusion")
    model2 = DualBranchModel(fusion="concat", spatial_pretrained=False).to(device)
    model2.eval()
    with torch.no_grad():
        out2 = model2(spatial, freq)
    print(f"  logits : {out2['logits'].shape}")
    assert out2["logits"].shape == (B, 2)

    # ── Test 3: Ablation — spatial only ───────────────────────────────────
    print("\nTest 3 — Ablation: spatial-only branch")
    model.set_branch_mode("spatial")
    with torch.no_grad():
        out3 = model(spatial, freq)
    # freq_feat should be all zeros
    assert out3["freq_feat"].abs().sum() == 0.0, "freq_feat should be zeroed"
    print(f"  logits : {out3['logits'].shape}  (freq branch zeroed)")

    # ── Test 4: Ablation — freq only ──────────────────────────────────────
    print("\nTest 4 — Ablation: freq-only branch")
    model.set_branch_mode("freq")
    with torch.no_grad():
        out4 = model(spatial, freq)
    assert out4["spatial_feat"].abs().sum() == 0.0, "spatial_feat should be zeroed"
    print(f"  logits : {out4['logits'].shape}  (spatial branch zeroed)")

    # ── Test 5: Gradient flow ─────────────────────────────────────────────
    print("\nTest 5 — Gradient flow (full model)")
    model5 = DualBranchModel(spatial_pretrained=False, branch="both").to(device)
    model5.train()
    out5   = model5(spatial, freq)
    labels = torch.randint(0, 2, (B,)).to(device)
    loss   = nn.CrossEntropyLoss()(out5["logits"], labels)
    loss.backward()
    print(f"  Loss : {loss.item():.4f}   Grads: OK")

    # ── Test 6: build_model() factory ─────────────────────────────────────
    print("\nTest 6 — build_model() factory")
    cfg = {
        "model": {
            "feature_dim":        256,
            "freq_channels":      [32, 64, 128],
            "dropout":            0.3,
            "fusion":             "attention_gate",
            "spatial_pretrained": False,
        }
    }
    model6 = build_model(cfg, branch="both").to(device)
    model6.eval()
    with torch.no_grad():
        out6 = model6(spatial, freq)
    assert out6["logits"].shape == (B, 2)
    print(f"  logits : {out6['logits'].shape}   OK")

    print("\n  dual_branch.py — all tests passed!")
