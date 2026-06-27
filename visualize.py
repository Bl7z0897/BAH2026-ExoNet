"""
Attention visualization + ablation results plot
BAH2026 | Tejeswin R | Phase 1 deliverable

Produces:
  - attention_heatmap.png  : attention weights overlaid on light curve
  - ablation_comparison.png: CNN vs CNN+Attention AUC/F1 bar chart
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

from model import ExoNetAttention, ExoNetCNNBaseline
from train import make_synthetic_dataset

# -----------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
})

PURPLE = "#534AB7"
TEAL   = "#0F6E56"
CORAL  = "#993C1D"
GRAY   = "#888780"

# -----------------------------------------------------------------------
# Load checkpoints
# -----------------------------------------------------------------------
device = torch.device("cpu")

model_attn = ExoNetAttention(seq_len=201, n_heads=4, dropout=0.3)
ckpt_attn  = torch.load("checkpoints/exonet_attention.pt", map_location=device, weights_only=False)
model_attn.load_state_dict(ckpt_attn["model_state"])
model_attn.eval()

model_cnn = ExoNetCNNBaseline(dropout=0.3)
ckpt_cnn  = torch.load("checkpoints/cnn_baseline.pt", map_location=device, weights_only=False)
model_cnn.load_state_dict(ckpt_cnn["model_state"])
model_cnn.eval()

# -----------------------------------------------------------------------
# Figure 1: Attention heatmap on a planet + non-planet example
# -----------------------------------------------------------------------
X_val, y_val = make_synthetic_dataset(n_samples=600, seq_len=201,
                                       pos_fraction=0.15, seed=1)

# pick first confirmed planet and first non-planet
pc_idx  = (y_val == 1).nonzero(as_tuple=True)[0][0].item()
np_idx  = (y_val == 0).nonzero(as_tuple=True)[0][0].item()

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
fig.suptitle("ExoNet-Attention — transit detection & attention weights",
             fontsize=13, fontweight="normal", y=1.01)

for row, (idx, label_str) in enumerate([(pc_idx, "Planet Candidate (PC=1)"),
                                          (np_idx, "Non-Planet (NP=0)")]):
    xb = X_val[idx:idx+1]   # (1, 1, 201)

    with torch.no_grad():
        prob   = model_attn(xb).item()
        attn_w = model_attn.get_attention_weights()   # (1, 4, 50, 50)

    flux = xb.squeeze().numpy()
    t    = np.arange(len(flux))

    # Attention: mean over heads, then mean over query dimension -> (50,)
    attn_mean = attn_w[0].mean(dim=0).mean(dim=0).numpy()  # (50,)
    # Upsample to seq_len for overlay
    attn_up   = np.interp(t, np.linspace(0, len(t)-1, len(attn_mean)), attn_mean)
    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-9)

    # ---- left: flux with attention overlay ----
    ax = axes[row, 0]
    ax.plot(t, flux, color=TEAL if row==0 else GRAY,
            linewidth=1.2, label="flux", zorder=2)
    ax2 = ax.twinx()
    ax2.fill_between(t, attn_norm, alpha=0.25,
                     color=PURPLE if row==0 else CORAL, label="attention")
    ax2.set_ylim(0, 3)
    ax2.set_ylabel("attention (norm)", fontsize=9, color=PURPLE if row==0 else CORAL)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("phase bin")
    ax.set_ylabel("normalized flux")
    ax.set_title(f"{label_str}\nP(planet) = {prob:.3f}", fontsize=10)

    # ---- right: attention matrix heatmap (head 0) ----
    ax = axes[row, 1]
    head0 = attn_w[0, 0].numpy()     # (50, 50)
    im = ax.imshow(head0, aspect="auto", cmap="Blues" if row==0 else "Oranges",
                   vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Attention matrix — head 0\n{label_str[:20]}...", fontsize=10)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")

plt.tight_layout()
plt.savefig("attention_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved: attention_heatmap.png")

# -----------------------------------------------------------------------
# Figure 2: Ablation comparison bar chart
# -----------------------------------------------------------------------
with open("checkpoints/training_history.json") as f:
    hist_attn = json.load(f)

best_attn = max(hist_attn, key=lambda r: r["auc"])
best_cnn  = {"auc": ckpt_cnn["val_auc"], "val_f1": ckpt_cnn["val_f1"]}

metrics_labels = ["Val AUC", "Val F1"]
cnn_vals  = [best_cnn["auc"],         best_cnn["val_f1"]]
attn_vals = [best_attn["auc"],        best_attn["f1"]]

x     = np.arange(len(metrics_labels))
width = 0.32

fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar(x - width/2, cnn_vals,  width, label="CNN-only (ablation)",
               color=GRAY,   alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, attn_vals, width, label="CNN + Attention (proposed)",
               color=PURPLE, alpha=0.85, edgecolor="white")

ax.set_ylim(0, 1.12)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=12)
ax.set_ylabel("Score")
ax.set_title("Ablation B: CNN-only vs CNN + Multi-Head Attention\n(synthetic data, phase 1)",
             fontsize=11)
ax.legend(fontsize=10)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.text(0.98, 0.05,
        "Note: synthetic data — rerun after\nShreya's real Kepler pipeline",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=CORAL,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=CORAL, alpha=0.7))

plt.tight_layout()
plt.savefig("ablation_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: ablation_comparison.png")

# -----------------------------------------------------------------------
# Figure 3: Training curves
# -----------------------------------------------------------------------
epochs_attn = [r["epoch"] for r in hist_attn]
auc_attn    = [r["auc"]   for r in hist_attn]
loss_attn   = [r["val_loss"] for r in hist_attn]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(epochs_attn, auc_attn, color=PURPLE, linewidth=2, label="CNN+Attention")
ax1.axhline(best_cnn["auc"], color=GRAY, linewidth=1.5,
            linestyle="--", label=f"CNN-only best ({best_cnn['auc']:.4f})")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Val AUC")
ax1.set_title("Validation AUC over training")
ax1.set_ylim(0.5, 1.05)
ax1.legend()

ax2.plot(epochs_attn, loss_attn, color=TEAL, linewidth=2, label="CNN+Attention val loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Loss")
ax2.set_title("Validation loss over training")
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png", bbox_inches="tight")
plt.close()
print("  Saved: training_curves.png")
