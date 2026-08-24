"""
models/freq_branch.py — Lightweight CNN frequency branch
Processes 2D FFT log-magnitude spectra → feature vector for fusion

Input  : (B, 1, 224, 224)  single-channel log-magnitude FFT spectrum
Output : (B, feature_dim)  feature vector  [extract_features=True]
         (B, 2)             logits          [extract_features=False]

Architecture
────────────
The spectrum is processed by a stack of ConvBlocks whose channel widths
are controlled by `freq_channels` (default [32, 64, 128] from config).
Each block:   Conv2d → BN → ReLU → MaxPool(2×2)

After the conv stack, GlobalAveragePooling collapses spatial dimensions,
and a small projection head maps to `feature_dim` (matching the spatial
branch) so both streams can be fused on equal footing.

Design rationale
────────────────
• Lightweight by design — frequency artifacts are complementary cues,
  not the primary signal, so we keep this branch fast and low-memory.
• Single-channel input — the FFT spectrum is computed from the grayscale
  version of the face patch (see dataset.compute_fft).
• No pretrained weights — there is no ImageNet-equivalent for FFT
  spectra, so this branch is always trained from scratch.
• Mirrors the SpatialBranch API (extract_features / standalone modes)
  so the two branches are drop-in interchangeable for ablation studies.
"""

import torch
import torch.nn as nn
from typing import List


# ─────────────────────────────────────────────
# Building block
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2d(3×3) → BatchNorm → ReLU → MaxPool(2×2)
    Standard spectral feature extractor block.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────
# Frequency Branch
# ─────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Lightweight CNN that classifies deepfakes from FFT magnitude spectra.

    Args:
        freq_channels    : list of output channel widths for each ConvBlock
                           e.g. [32, 64, 128]  (from config.model.freq_channels)
        feature_dim      : output feature vector size (must match SpatialBranch)
        dropout          : dropout rate before classifier / projection head
        extract_features : True  → return (B, feature_dim) for dual-branch fusion
                           False → return (B, 2) logits for ablation / standalone
    """

    def __init__(
        self,
        freq_channels:    List[int] = None,
        feature_dim:      int       = 256,
        dropout:          float     = 0.3,
        extract_features: bool      = True,
    ):
        super().__init__()
        if freq_channels is None:
            freq_channels = [32, 64, 128]

        self.extract_features = extract_features
        self.feature_dim      = feature_dim

        # ── Convolutional stack ───────────────────────────────────────────
        # Build dynamically from freq_channels list.
        # Spatial resolution halves at each MaxPool:
        #   224 → 112 → 56 → 28  (for default 3 blocks)
        conv_layers = []
        in_ch = 1                          # single-channel FFT input
        for out_ch in freq_channels:
            conv_layers.append(ConvBlock(in_ch, out_ch, pool=True))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*conv_layers)
        # Final channel count after the stack
        self._out_ch = freq_channels[-1]   # e.g. 128

        # ── Global average pooling ────────────────────────────────────────
        # Collapses (B, C, H', W') → (B, C) regardless of spatial size
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Projection head ───────────────────────────────────────────────
        # Maps conv output → feature_dim (same dim as SpatialBranch output)
        self.feature_head = nn.Sequential(
            nn.Linear(self._out_ch, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Standalone classification head ────────────────────────────────
        # Only used in ablation (extract_features=False)
        self.classifier = nn.Linear(feature_dim, 2)

        # Weight initialisation
        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_weights(self):
        """Kaiming init for conv layers, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W)  — log-magnitude FFT spectrum, values in [0, 1]

        Returns:
            features : (B, feature_dim)  if extract_features=True
            logits   : (B, 2)            if extract_features=False
        """
        x = self.conv_stack(x)        # (B, C_last, H', W')
        x = self.gap(x)               # (B, C_last, 1, 1)
        x = x.flatten(1)              # (B, C_last)
        x = self.feature_head(x)      # (B, feature_dim)

        if self.extract_features:
            return x                  # → goes to fusion module
        return self.classifier(x)     # → standalone logits

    # ── Utilities ─────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def feature_map_sizes(self, input_size: int = 224) -> list:
        """
        Return spatial sizes at each conv block output (for debugging).
        Assumes square input of `input_size`.
        """
        sizes = []
        s = input_size
        for _ in self.conv_stack:
            s = s // 2          # each block has MaxPool(2×2)
            sizes.append(s)
        return sizes


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Test 1: Feature extraction mode (used in dual-branch) ─────────────
    print("Test 1 — Feature extraction mode")
    model = FrequencyBranch(
        freq_channels=[32, 64, 128],
        feature_dim=256,
        dropout=0.3,
        extract_features=True,
    ).to(device)
    model.eval()

    x = torch.randn(4, 1, 224, 224).to(device)   # (B, 1, H, W) FFT input
    with torch.no_grad():
        out = model(x)

    print(f"  Input  : {x.shape}")
    print(f"  Output : {out.shape}   ← feature vector for fusion")
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"

    params = model.count_parameters()
    print(f"  Params : {params['total']:,} total | "
          f"{params['trainable']:,} trainable")
    print(f"  Feature map sizes: {model.feature_map_sizes(224)}")

    # ── Test 2: Standalone classifier mode (ablation) ─────────────────────
    print("\nTest 2 — Standalone classifier mode")
    model2 = FrequencyBranch(
        freq_channels=[32, 64, 128],
        feature_dim=256,
        extract_features=False,
    ).to(device)
    model2.eval()

    with torch.no_grad():
        logits = model2(x)

    print(f"  Input  : {x.shape}")
    print(f"  Output : {logits.shape}   ← logits (real vs fake)")
    assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"

    # ── Test 3: Custom channel config ─────────────────────────────────────
    print("\nTest 3 — Custom channel config [16, 32, 64, 128]")
    model3 = FrequencyBranch(
        freq_channels=[16, 32, 64, 128],
        feature_dim=256,
        extract_features=True,
    ).to(device)
    model3.eval()

    with torch.no_grad():
        out3 = model3(x)

    print(f"  Output : {out3.shape}")
    print(f"  Feature map sizes: {model3.feature_map_sizes(224)}")
    assert out3.shape == (4, 256)

    # ── Test 4: Gradient flow ──────────────────────────────────────────────
    print("\nTest 4 — Gradient flow")
    model4 = FrequencyBranch(extract_features=False).to(device)
    model4.train()
    x_grad = torch.randn(4, 1, 224, 224).to(device)
    labels = torch.randint(0, 2, (4,)).to(device)
    logits = model4(x_grad)
    loss   = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    print(f"  Loss   : {loss.item():.4f}")
    print(f"  Grads  : OK")

    # ── Test 5: Input size invariance (thanks to AdaptiveAvgPool) ──────────
    print("\nTest 5 — Input size invariance")
    for sz in [112, 224, 256]:
        x_sz = torch.randn(2, 1, sz, sz).to(device)
        with torch.no_grad():
            o = model(x_sz)
        print(f"  Input {sz}×{sz} → Output {o.shape}")
        assert o.shape == (2, 256)

    print("\n  freq_branch.py — all tests passed!")
