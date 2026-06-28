"""
Training loop for ExoNet-Attention — REAL KEPLER DATA VERSION
BAH2026 | Tejeswin R
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
import json
from pathlib import Path

from model import ExoNetAttention, ExoNetCNNBaseline


def compute_metrics(all_probs, all_labels, threshold=0.5):
    preds = (np.array(all_probs) >= threshold).astype(int)
    labs  = np.array(all_labels).astype(int)
    auc   = roc_auc_score(labs, all_probs) if len(np.unique(labs)) > 1 else 0.5
    f1    = f1_score(labs, preds, zero_division=0)
    acc   = (preds == labs).mean()
    return {"auc": auc, "f1": f1, "acc": acc}


def train(model, train_loader, val_loader, pos_weight, device,
          lr=1e-3, weight_decay=1e-4, n_epochs=50, patience=10,
          save_path="best_model.pt", run_name="exonet"):

    model.to(device)
    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_auc = 0.0
    patience_ctr = 0
    history      = []

    print(f"\n  Training {run_name} on {device}")
    print(f"  Epochs: {n_epochs}  |  Patience: {patience}  |  LR: {lr}")
    print("-" * 60)
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val AUC':>8}  {'Val F1':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if model.training:
                    xb = xb + torch.randn_like(xb) * 0.002
                probs = model(xb)
                w     = torch.where(yb == 1, pos_weight.to(device), torch.ones_like(yb))
                loss  = (criterion(probs, yb) * w).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss   = 0.0
        all_probs  = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                probs  = model(xb)
                w      = torch.where(yb == 1, pos_weight.to(device), torch.ones_like(yb))
                loss   = (criterion(probs, yb) * w).mean()
                val_loss   += loss.item()
                all_probs  .extend(probs.cpu().numpy())
                all_labels .extend(yb.cpu().numpy())

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
        print(f"  {epoch:>5}  {train_loss:>10.5f}  {val_loss:>9.5f}  {metrics['auc']:>8.4f}  {metrics['f1']:>7.4f}  {elapsed:>5.1f}s")

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
                print(f"\n  Early stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                break

    print("-" * 60)
    print(f"  Best val AUC: {best_val_auc:.4f}  |  Saved to: {save_path}")
    return history


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEQ_LEN    = 201
    BATCH_SIZE = 64
    N_EPOCHS   = 100
    PATIENCE   = 20
    LR         = 1e-4
    DATA_PATH  = r"E:\ISRO Exoplanet Hackthon\Phase One\Sara\sara\kepler_dr25_synthetic_2000_batch.npz"

    print("  Loading real Kepler data...")
    data  = np.load(DATA_PATH)
    X_all = torch.tensor(data['flux'],   dtype=torch.float32)
    y_all = torch.tensor(data['labels'], dtype=torch.float32)

    idx_train, idx_val = train_test_split(
        range(len(y_all)), test_size=0.2, stratify=y_all.numpy(), random_state=42
    )

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val,   y_val   = X_all[idx_val],   y_all[idx_val]

    print(f"  Train: {X_train.shape}  |  Planets: {y_train.sum().int()}")
    print(f"  Val:   {X_val.shape}    |  Planets: {y_val.sum().int()}")

    n_pos      = y_train.sum().item()
    n_neg      = len(y_train) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    print(f"  pos_weight: {pos_weight:.2f}  ({n_pos:.0f} PC vs {n_neg:.0f} NP)")

    train_ds     = TensorDataset(X_train, y_train)
    val_ds       = TensorDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    Path("checkpoints").mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  ABLATION B: CNN-only baseline")
    cnn_baseline = ExoNetCNNBaseline(dropout=0.5)
    train(cnn_baseline, train_loader, val_loader,
          pos_weight=pos_weight, device=device,
          lr=LR, n_epochs=N_EPOCHS, patience=PATIENCE,
          save_path="checkpoints/cnn_baseline.pt",
          run_name="ExoNet-CNN (baseline)")

    print("\n" + "=" * 60)
    print("  PROPOSED: CNN + Multi-Head Attention")
    model   = ExoNetAttention(seq_len=SEQ_LEN, n_heads=4, dropout=0.5)
    history = train(model, train_loader, val_loader,
                    pos_weight=pos_weight, device=device,
                    lr=LR, n_epochs=N_EPOCHS, patience=PATIENCE,
                    save_path="checkpoints/exonet_attention.pt",
                    run_name="ExoNet-Attention (proposed)")

    with open("checkpoints/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n  Done. Checkpoints saved.")
