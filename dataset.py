"""
dataset.py — PyTorch Dataset for deepfake detection
Supports: dummy synthetic data (for pipeline testing) and real DFDC / FF++ data
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 224):
    """
    Returns albumentations transform pipeline.
    split: 'train' | 'val' | 'test'
    """
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ImageCompression(quality_range=(60, 100), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────
# FFT helper
# ─────────────────────────────────────────────

def compute_fft(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute log-magnitude FFT spectrum from an RGB tensor.
    img_tensor : (C, H, W) float32 — already normalized
    returns    : (1, H, W) float32 — log magnitude spectrum in [0, 1]
    """
    gray    = img_tensor.mean(dim=0)                      # (H, W)
    fft     = torch.fft.fft2(gray)                        # complex (H, W)
    fft     = torch.fft.fftshift(fft)                     # center low-freq
    mag     = torch.abs(fft)                              # magnitude
    log_mag = torch.log1p(mag)                            # log scale
    # normalize to [0, 1]
    lo, hi  = log_mag.min(), log_mag.max()
    log_mag = (log_mag - lo) / (hi - lo + 1e-8)
    return log_mag.unsqueeze(0)                           # (1, H, W)


# ─────────────────────────────────────────────
# Dummy Dataset  (synthetic — no real data needed)
# ─────────────────────────────────────────────

class DummyDeepfakeDataset(Dataset):
    """
    Generates random face-sized RGB images labelled real (0) or fake (1).
    Used to verify the full pipeline before real data arrives.
    """

    def __init__(self, num_samples: int = 200, image_size: int = 224,
                 split: str = "train", seed: int = 42):
        self.num_samples = num_samples
        self.image_size  = image_size
        self.transform   = get_transforms(split, image_size)
        rng              = np.random.default_rng(seed)
        self.labels      = rng.integers(0, 2, size=num_samples).tolist()
        # fake images get slightly different pixel stats to mimic real differences
        self._rng        = rng

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.labels[idx]

        # Real: random natural-ish image; Fake: add subtle high-freq noise
        img = self._rng.integers(60, 200, (self.image_size, self.image_size, 3),
                                  dtype=np.uint8)
        if label == 1:                                    # fake — add HF artifact
            noise = self._rng.normal(0, 15,
                                      (self.image_size, self.image_size, 3))
            img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        augmented   = self.transform(image=img)
        spatial     = augmented["image"]                  # (3, H, W)
        freq        = compute_fft(spatial)                # (1, H, W)

        return {
            "spatial": spatial,
            "freq":    freq,
            "label":   torch.tensor(label, dtype=torch.long),
            "path":    f"dummy_{idx}.jpg"
        }


# ─────────────────────────────────────────────
# Real Dataset  (DFDC / FF++)
# ─────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Loads real/fake face images from a flat folder structure:

        data/
          dfdc/
            real/   *.jpg / *.png
            fake/   *.jpg / *.png
          faceforensics/
            real/
            fake/

    Pass a list of (image_path, label) tuples via `samples`.
    Use build_samples() to generate this list from a root folder.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, samples: list, image_size: int = 224, split: str = "train"):
        self.samples    = samples          # list of (path_str, int_label)
        self.transform  = get_transforms(split, image_size)
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        if img is None:
            # Return a blank image if file is corrupt — keeps training stable
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        spatial   = augmented["image"]           # (3, H, W)
        freq      = compute_fft(spatial)         # (1, H, W)

        return {
            "spatial": spatial,
            "freq":    freq,
            "label":   torch.tensor(label, dtype=torch.long),
            "path":    path
        }


def build_samples(root: str, max_per_class: int = None) -> list:
    """
    Scan root/real/ and root/fake/ folders.
    Returns list of (path, label) — real=0, fake=1.
    """
    samples = []
    for label, subfolder in enumerate(["real", "fake"]):
        folder = Path(root) / subfolder
        if not folder.exists():
            print(f"  [warn] folder not found: {folder}")
            continue
        files = [f for f in folder.iterdir()
                 if f.suffix.lower() in DeepfakeDataset.EXTENSIONS]
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            samples.append((str(f), label))
    random.shuffle(samples)
    return samples


def split_samples(samples: list, ratios=(0.70, 0.15, 0.15)):
    """Split sample list into train / val / test."""
    n      = len(samples)
    t      = int(n * ratios[0])
    v      = int(n * ratios[1])
    return samples[:t], samples[t:t+v], samples[t+v:]


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(cfg: dict, use_dummy: bool = False):
    """
    Returns (train_loader, val_loader, test_loader).
    Set use_dummy=True to use synthetic data while real data downloads.
    """
    if use_dummy:
        print("  [DataLoader] Using DUMMY synthetic dataset")
        train_ds = DummyDeepfakeDataset(num_samples=320, split="train")
        val_ds   = DummyDeepfakeDataset(num_samples=80,  split="val",   seed=1)
        test_ds  = DummyDeepfakeDataset(num_samples=80,  split="test",  seed=2)
    else:
        dataset  = cfg["data"]["dataset"]          # "dfdc" | "ff++" | "both"
        roots    = []
        if dataset in ("dfdc", "both"):
            roots.append(cfg["data"]["dfdc_root"])
        if dataset in ("ff++", "both"):
            roots.append(cfg["data"]["ff_root"])

        all_samples = []
        for root in roots:
            all_samples += build_samples(root)

        if len(all_samples) == 0:
            raise ValueError("No images found. Check your data folder structure.")

        train_s, val_s, test_s = split_samples(
            all_samples, ratios=cfg["data"]["split_ratios"])

        img_size = cfg["preprocess"]["face_size"]
        train_ds = DeepfakeDataset(train_s, img_size, split="train")
        val_ds   = DeepfakeDataset(val_s,   img_size, split="val")
        test_ds  = DeepfakeDataset(test_s,  img_size, split="test")

    bs  = cfg["train"]["batch_size"]
    nw  = cfg["data"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              drop_last=True)   # prevents BN1d crash on size-1 tail batch
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    print(f"  [DataLoader] Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing DummyDeepfakeDataset...")

    ds = DummyDeepfakeDataset(num_samples=8, split="train")
    sample = ds[0]

    print(f"  spatial shape : {sample['spatial'].shape}")
    print(f"  freq shape    : {sample['freq'].shape}")
    print(f"  label         : {sample['label']}")
    print(f"  spatial range : [{sample['spatial'].min():.2f}, "
          f"{sample['spatial'].max():.2f}]")
    print(f"  freq range    : [{sample['freq'].min():.2f}, "
          f"{sample['freq'].max():.2f}]")

    # Test DataLoader
    cfg = {
        "data":       {"dataset": "dfdc", "dfdc_root": "data/dfdc",
                       "ff_root": "data/faceforensics",
                       "split_ratios": [0.70, 0.15, 0.15], "num_workers": 0},
        "train":      {"batch_size": 4},
        "preprocess": {"face_size": 224},
    }
    train_loader, val_loader, test_loader = get_dataloaders(cfg, use_dummy=True)
    batch = next(iter(train_loader))

    print(f"\n  Batch spatial : {batch['spatial'].shape}")
    print(f"  Batch freq    : {batch['freq'].shape}")
    print(f"  Batch labels  : {batch['label']}")
    print("\n  dataset.py self-test passed!")
