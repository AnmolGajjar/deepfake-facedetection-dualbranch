"""
train.py — Full training loop for the Dual-Branch Deepfake Detector

Usage
─────
    python train.py                          # train with config.yaml defaults
    python train.py --config config.yaml     # explicit config path
    python train.py --branch spatial         # spatial-only ablation
    python train.py --branch freq            # freq-only ablation
    python train.py --dummy                  # quick smoke-test on synthetic data

What this script does
─────────────────────
  1. Loads config.yaml
  2. Builds DualBranchModel (or single-branch variant for ablation)
  3. Runs train / val / test loops with:
       • AMP (automatic mixed precision)
       • Cosine LR schedule with linear warm-up
       • Early stopping on val AUC
       • Best-model checkpointing
  4. Logs per-epoch metrics (loss, accuracy, AUC, F1) to
       results/<run_name>/metrics.csv  and  metrics.json
  5. Saves final test-set results to  results/<run_name>/test_results.json
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataset import get_dataloaders
from models import DualBranchModel, build_model


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# LR scheduler with warm-up
# ─────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """
    Linear warm-up for `warmup_epochs`, then cosine decay for the remainder.
    Both sub-schedulers operate on epoch boundaries (not step boundaries).
    """
    warmup  = cfg["train"]["warmup_epochs"]
    epochs  = cfg["train"]["epochs"]
    decay   = max(epochs - warmup, 1)

    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=decay,
        eta_min=cfg["train"]["lr"] * 1e-3,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup],
    )


# ─────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────

def compute_metrics(
    all_labels:  list,
    all_probs:   list,
    threshold:   float = 0.5,
) -> dict:
    """
    Compute accuracy, AUC, and F1 from accumulated labels and probabilities.

    all_labels : list of int (0=real, 1=fake)
    all_probs  : list of float — P(fake) for each sample
    """
    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        # Only one class present — can happen in tiny batches
        auc = 0.5

    return {"accuracy": acc, "auc": auc, "f1": f1}


# ─────────────────────────────────────────────
# Results logger
# ─────────────────────────────────────────────

class ResultsLogger:
    """
    Writes per-epoch metrics to CSV and accumulates a JSON summary.
    """

    def __init__(self, results_dir: str, run_name: str):
        self.run_dir = Path(results_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path  = self.run_dir / "metrics.csv"
        self.json_path = self.run_dir / "metrics.json"
        self.history   = []

        # Write CSV header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "phase",
                "loss", "accuracy", "auc", "f1", "lr",
            ])

    def log(self, epoch: int, phase: str, loss: float,
            metrics: dict, lr: float):
        row = {
            "epoch":    epoch,
            "phase":    phase,
            "loss":     round(loss, 6),
            "accuracy": round(metrics.get("accuracy", 0), 6),
            "auc":      round(metrics.get("auc", 0), 6),
            "f1":       round(metrics.get("f1", 0), 6),
            "lr":       lr,
        }
        self.history.append(row)

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row[k] for k in
                             ["epoch", "phase", "loss",
                              "accuracy", "auc", "f1", "lr"]])

        # Overwrite JSON with full history
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def save_test_results(self, results: dict):
        path = self.run_dir / "test_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Logger] Test results → {path}")


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the full train / validate / test lifecycle.

    Args:
        model       : DualBranchModel instance
        cfg         : full config dict
        device      : torch.device
        run_name    : identifier for this run (used for file paths)
    """

    def __init__(
        self,
        model:    DualBranchModel,
        cfg:      dict,
        device:   torch.device,
        run_name: str = "run",
    ):
        self.model   = model.to(device)
        self.cfg     = cfg
        self.device  = device
        self.run_name = run_name

        t = cfg["train"]

        # ── Optimizer ────────────────────────────────────────────────────
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=t["lr"],
            weight_decay=t["weight_decay"],
        )

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss()

        # ── AMP ──────────────────────────────────────────────────────────
        self.use_amp = t.get("amp", True) and device.type == "cuda"
        self.scaler  = GradScaler("cuda", enabled=self.use_amp)

        # ── Early stopping ────────────────────────────────────────────────
        self.patience       = t.get("early_stopping", 7)
        self.best_val_auc   = 0.0
        self.epochs_no_improve = 0

        # ── Checkpoint dir ────────────────────────────────────────────────
        self.ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt = self.ckpt_dir / f"{run_name}_best.pt"

        # ── Logger ────────────────────────────────────────────────────────
        self.logger = ResultsLogger(
            results_dir=cfg["logging"]["results_dir"],
            run_name=run_name,
        )

        # ── Threshold ─────────────────────────────────────────────────────
        self.threshold = cfg["eval"].get("threshold", 0.5)

    # ── One training epoch ────────────────────────────────────────────────

    def train_one_epoch(self, loader, epoch: int) -> tuple:
        """
        Run one full pass over the training set.
        Returns (avg_loss, metrics_dict).
        """
        self.model.train()

        total_loss  = 0.0
        all_labels  = []
        all_probs   = []

        for batch in loader:
            spatial = batch["spatial"].to(self.device, non_blocking=True)
            freq    = batch["freq"].to(self.device,    non_blocking=True)
            labels  = batch["label"].to(self.device,   non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                out  = self.model(spatial, freq)
                loss = self.criterion(out["logits"], labels)

            self.scaler.scale(loss).backward()
            # Gradient clipping — prevents instability in early training
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)

            # Collect probabilities for metric computation
            probs = torch.softmax(out["logits"], dim=1)[:, 1]  # P(fake)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

        n        = len(all_labels)
        avg_loss = total_loss / max(n, 1)
        metrics  = compute_metrics(all_labels, all_probs, self.threshold)

        return avg_loss, metrics

    # ── Validation / test pass ────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> tuple:
        """
        Run one full pass over val or test set (no gradients).
        Returns (avg_loss, metrics_dict, attention_weights_mean).
        """
        self.model.eval()

        total_loss   = 0.0
        all_labels   = []
        all_probs    = []
        attn_spatial = []   # mean α (spatial weight) per batch
        attn_freq    = []   # mean β (freq weight) per batch

        for batch in loader:
            spatial = batch["spatial"].to(self.device, non_blocking=True)
            freq    = batch["freq"].to(self.device,    non_blocking=True)
            labels  = batch["label"].to(self.device,   non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                out  = self.model(spatial, freq)
                loss = self.criterion(out["logits"], labels)

            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(out["logits"], dim=1)[:, 1]
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            # Track mean attention weights across the batch
            w = out["attention_weights"].mean(dim=0).cpu().tolist()
            attn_spatial.append(w[0])
            attn_freq.append(w[1])

        n        = len(all_labels)
        avg_loss = total_loss / max(n, 1)
        metrics  = compute_metrics(all_labels, all_probs, self.threshold)
        metrics["mean_attn_spatial"] = float(np.mean(attn_spatial))
        metrics["mean_attn_freq"]    = float(np.mean(attn_freq))

        return avg_loss, metrics

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_auc: float, path: Path):
        torch.save({
            "epoch":      epoch,
            "val_auc":    val_auc,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "branch":     self.model.branch,
            "fusion":     self.model.fusion_type,
        }, path)

    def load_best_checkpoint(self):
        if self.best_ckpt.exists():
            ckpt = torch.load(self.best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            print(f"  [Trainer] Loaded best checkpoint "
                  f"(epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f})")
        else:
            print("  [Trainer] No checkpoint found — using current weights")

    # ── Main training loop ────────────────────────────────────────────────

    def fit(self, train_loader, val_loader):
        """
        Full training loop with warm-up, cosine decay, early stopping.
        """
        cfg     = self.cfg
        epochs  = cfg["train"]["epochs"]

        scheduler = build_scheduler(
            self.optimizer, cfg,
            steps_per_epoch=len(train_loader),
        )

        # Optionally freeze backbone for first few epochs to stabilise training
        warmup_freeze = cfg["train"].get("warmup_freeze_backbone", False)
        if warmup_freeze:
            print("  [Trainer] Freezing EfficientNet backbone during warm-up")
            self.model.freeze_spatial_backbone(True)

        print(f"\n{'─'*60}")
        print(f"  Training: {epochs} epochs | "
              f"device={self.device} | AMP={self.use_amp}")
        print(f"  Patience: {self.patience} epochs | "
              f"Branch: '{self.model.branch}'")
        print(f"{'─'*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Unfreeze backbone after warm-up
            if warmup_freeze and epoch == cfg["train"]["warmup_epochs"] + 1:
                print("  [Trainer] Unfreezing EfficientNet backbone")
                self.model.freeze_spatial_backbone(False)
                # Re-create optimizer to include newly unfrozen params
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=cfg["train"]["lr"] * 0.1,
                    weight_decay=cfg["train"]["weight_decay"],
                )

            # ── Train ────────────────────────────────────────────────────
            train_loss, train_metrics = self.train_one_epoch(train_loader, epoch)

            # ── Validate ─────────────────────────────────────────────────
            val_loss, val_metrics = self.evaluate(val_loader)

            # ── LR step ──────────────────────────────────────────────────
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            elapsed = time.time() - t0

            # ── Print ─────────────────────────────────────────────────────
            print(
                f"Epoch {epoch:3d}/{epochs} "
                f"[{elapsed:5.1f}s]  "
                f"loss {train_loss:.4f} → {val_loss:.4f}  |  "
                f"acc {train_metrics['accuracy']:.3f} → {val_metrics['accuracy']:.3f}  |  "
                f"AUC {train_metrics['auc']:.3f} → {val_metrics['auc']:.3f}  |  "
                f"F1 {train_metrics['f1']:.3f} → {val_metrics['f1']:.3f}  |  "
                f"α={val_metrics['mean_attn_spatial']:.2f} "
                f"β={val_metrics['mean_attn_freq']:.2f}  "
                f"lr={current_lr:.2e}"
            )

            # ── Log ───────────────────────────────────────────────────────
            self.logger.log(epoch, "train", train_loss, train_metrics, current_lr)
            self.logger.log(epoch, "val",   val_loss,   val_metrics,   current_lr)

            # ── Checkpoint & early stopping ───────────────────────────────
            val_auc = val_metrics["auc"]
            if val_auc > self.best_val_auc:
                self.best_val_auc      = val_auc
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_auc, self.best_ckpt)
                print(f"           ✓ New best val AUC: {val_auc:.4f} "
                      f"— checkpoint saved")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"\n  [EarlyStopping] No improvement for "
                          f"{self.patience} epochs — stopping at epoch {epoch}")
                    break

        print(f"\n  Training complete. Best val AUC: {self.best_val_auc:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────

    def test(self, test_loader) -> dict:
        """
        Load best checkpoint and evaluate on the held-out test set.
        Saves results to results/<run_name>/test_results.json.
        """
        print("\n  Loading best checkpoint for final test evaluation...")
        self.load_best_checkpoint()

        test_loss, test_metrics = self.evaluate(test_loader)

        results = {
            "run_name":  self.run_name,
            "branch":    self.model.branch,
            "test_loss": round(test_loss, 6),
            **{k: round(v, 6) for k, v in test_metrics.items()},
        }

        print(f"\n{'─'*60}")
        print(f"  TEST RESULTS  ({self.run_name})")
        print(f"{'─'*60}")
        print(f"  Loss     : {test_loss:.4f}")
        print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"  AUC      : {test_metrics['auc']:.4f}")
        print(f"  F1       : {test_metrics['f1']:.4f}")
        print(f"  Attn α   : {test_metrics['mean_attn_spatial']:.4f} (spatial)")
        print(f"  Attn β   : {test_metrics['mean_attn_freq']:.4f} (freq)")
        print(f"{'─'*60}\n")

        self.logger.save_test_results(results)
        return results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train dual-branch deepfake detector")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config YAML (default: config.yaml)")
    p.add_argument("--branch", default="both",
                   choices=["both", "spatial", "freq"],
                   help="Branch mode: both | spatial | freq")
    p.add_argument("--dummy", action="store_true",
                   help="Use synthetic dummy data (smoke-test / no real data needed)")
    p.add_argument("--run-name", default=None,
                   help="Override run name for results/checkpoint dirs")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Seed ─────────────────────────────────────────────────────────────
    set_seed(cfg["train"]["seed"])

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── Run name ─────────────────────────────────────────────────────────
    run_name = args.run_name or f"{args.branch}_branch"

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg, use_dummy=args.dummy
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg, branch=args.branch)
    params = model.count_parameters()
    print(f"  Model  : {params['total']['total']:,} total params "
          f"| {params['total']['trainable']:,} trainable")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model=model, cfg=cfg, device=device, run_name=run_name)
    trainer.fit(train_loader, val_loader)

    # ── Test ──────────────────────────────────────────────────────────────
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
