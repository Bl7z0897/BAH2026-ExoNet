"""
Training loop for ExoNet-Attention
BAH2026 | Tejeswin R | Phase 1 deliverable

Features:
  - Weighted CrossEntropy for class imbalance
  - Mixed precision (AMP) — GPU when available, CPU fallback
  - Early stopping on val AUC with patience
  - Per-epoch logging: loss, accuracy, AUC, F1
  - Saves best checkpoint as best_model.pt
  - Works on synthetic data if Shreya's real data not yet available
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import time
import json
from pathlib import Path

from model import ExoNetAttention, ExoNetCNNBaseline


# ---------------------------------------------------------------------------
# Synthetic dataset generator (placeholder until Shreya's pipeline is ready)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n_samples=2000, seq_len=201, pos_fraction=0.15,
                             seed=42):
    """
    Generate synthetic light curves with injected box transits.
    Mimics phase-folded Kepler photometry.

    Returns: flux (N, 1, L), labels (N,) as float32 tensors
    """
    rng = np.random.default_rng(seed)
    flux   = np.ones((n_samples, seq_len), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.float32)

    n_pos = int(n_samples * pos_fraction)
    pos_idx = rng.choice(n_samples, n_pos, replace=False)

    for i in pos_idx:
        # Random transit parameters
        depth    = rng.uniform(0.001, 0.02)        # 0.1% – 2% depth
        duration = rng.integers(5, 25)             # 5–25 timesteps
        center   = rng.integers(duration, seq_len - duration)

        # Box transit
        flux[i, center - duration//2 : center + duration//2] -= depth

        # Limb-darkening smear on ingress/egress (2 timesteps)
        flux[i, center - duration//2 - 2 : center - duration//2] -= depth * 0.5
        flux[i, center + duration//2     : center + duration//2 + 2] -= depth * 0.5

        labels[i] = 1.0

    # Add stellar variability noise
    flux += rng.normal(0, 0.001, flux.shape).astype(np.float32)    # photon noise
    flux += (rng.normal(0, 0.0005, (n_samples, 1))                  # per-star offset
             .astype(np.float32))

    flux_tensor   = torch.tensor(flux).unsqueeze(1)   # (N, 1, L)
    labels_tensor = torch.tensor(labels)
    return flux_tensor, labels_tensor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_probs, all_labels, threshold=0.5):
    preds = (np.array(all_probs) >= threshold).astype(int)
    labs  = np.array(all_labels).astype(int)
    auc   = roc_auc_score(labs, all_probs) if len(np.unique(labs)) > 1 else 0.5
    f1    = f1_score(labs, preds, zero_division=0)
    acc   = (preds == labs).mean()
    return {"auc": auc, "f1": f1, "acc": acc}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    pos_weight,
    device,
    lr=1e-3,
    weight_decay=1e-4,
    n_epochs=50,
    patience=10,
    save_path="best_model.pt",
    run_name="exonet",
):
    model.to(device)
    criterion = nn.BCELoss(reduction='none')   # we apply pos_weight manually
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # AMP scaler — graceful CPU fallback
    use_amp   = device.type == "cuda"
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc  = 0.0
    patience_ctr  = 0
    history       = []

    print(f"\n  Training {run_name} on {device}")
    print(f"  Epochs: {n_epochs}  |  Patience: {patience}  |  LR: {lr}")
    print("-" * 60)
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  "
          f"{'Val AUC':>8}  {'Val F1':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                probs = model(xb)
                # Weighted BCE: positive examples get higher weight
                w     = torch.where(yb == 1, pos_weight.to(device),
                                    torch.ones_like(yb))
                loss  = (criterion(probs, yb) * w).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ---- validate ----
        model.eval()
        val_loss   = 0.0
        all_probs  = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                probs  = model(xb)
                w      = torch.where(yb == 1, pos_weight.to(device),
                                     torch.ones_like(yb))
                loss   = (criterion(probs, yb) * w).mean()
                val_loss  += loss.item()
                all_probs .extend(probs.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_loss /= len(val_loader)
        metrics   = compute_metrics(all_probs, all_labels)
        elapsed   = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss":   round(val_loss,   5),
            **{k: round(v, 4) for k, v in metrics.items()},
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        history.append(row)

        print(f"  {epoch:>5}  {train_loss:>10.5f}  {val_loss:>9.5f}  "
              f"{metrics['auc']:>8.4f}  {metrics['f1']:>7.4f}  {elapsed:>5.1f}s")

        # ---- early stopping ----
        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            patience_ctr = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     best_val_auc,
                "val_f1":      metrics["f1"],
                "history":     history,
            }, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val AUC: {best_val_auc:.4f})")
                break

    print("-" * 60)
    print(f"  Best val AUC: {best_val_auc:.4f}  |  Saved to: {save_path}")
    return history


# ---------------------------------------------------------------------------
# Entry point — runs on synthetic data, ready to swap in Shreya's .npz
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ_LEN     = 201
    BATCH_SIZE  = 64
    N_EPOCHS    = 30       # reduced for demo; use 100 for real run
    PATIENCE    = 8
    LR          = 3e-4

    # ---- Load data ----
    # Swap this block with np.load("train.npz") when Shreya's data is ready:
    #   flux_train = torch.tensor(np.load("train.npz")["flux"])
    #   y_train    = torch.tensor(np.load("train.npz")["labels"])

    print("  Generating synthetic light curves (placeholder for Shreya's data)...")
    X_train, y_train = make_synthetic_dataset(n_samples=3000, seq_len=SEQ_LEN,
                                               pos_fraction=0.15, seed=0)
    X_val,   y_val   = make_synthetic_dataset(n_samples=600,  seq_len=SEQ_LEN,
                                               pos_fraction=0.15, seed=1)

    print(f"  Train: {X_train.shape}  |  Positives: {y_train.sum().int()}")
    print(f"  Val:   {X_val.shape}    |  Positives: {y_val.sum().int()}")

    # ---- Class imbalance ----
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    print(f"  pos_weight: {pos_weight:.2f}  (compensates {n_pos:.0f} PC vs {n_neg:.0f} NP)")

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ---- Ablation B: CNN only ----
    print("\n" + "=" * 60)
    print("  ABLATION B: CNN-only baseline")
    cnn_baseline = ExoNetCNNBaseline(dropout=0.3)
    train(cnn_baseline, train_loader, val_loader,
          pos_weight=pos_weight, device=device,
          lr=LR, n_epochs=N_EPOCHS, patience=PATIENCE,
          save_path="checkpoints/cnn_baseline.pt",
          run_name="ExoNet-CNN (baseline)")

    # ---- Proposed model: CNN + Attention ----
    print("\n" + "=" * 60)
    print("  PROPOSED: CNN + Multi-Head Attention")
    Path("checkpoints").mkdir(exist_ok=True)
    model = ExoNetAttention(seq_len=SEQ_LEN, n_heads=4, dropout=0.3)
    history = train(model, train_loader, val_loader,
                    pos_weight=pos_weight, device=device,
                    lr=LR, n_epochs=N_EPOCHS, patience=PATIENCE,
                    save_path="checkpoints/exonet_attention.pt",
                    run_name="ExoNet-Attention (proposed)")

    # Save history
    with open("checkpoints/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n  Phase 1 complete. Artifacts saved to checkpoints/")
    print("  Hand off to Phase 2: swap synthetic data for Shreya's train.npz")
