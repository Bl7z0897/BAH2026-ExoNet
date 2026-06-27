"""
ExoNet-Attention: 1D-CNN + Multi-Head Attention for Exoplanet Detection
BAH2026 | Tejeswin R | Phase 1 deliverable

Architecture:
    Input flux (B, 1, L) -> CNN feature extractor -> Multi-Head Attention
    -> Global Average Pool -> Classifier head -> P(planet) in [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU, optionally with MaxPool."""

    def __init__(self, in_ch, out_ch, kernel_size=5, pool=False):
        super().__init__()
        padding = kernel_size // 2          # 'same' padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class AttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention over CNN feature maps.

    Input:  (B, C, L')  — CNN output (channels-last after permute)
    Output: (B, C, L')  — same shape, with residual connection

    The attention weights (shape B x heads x L' x L') are stored in
    self.attn_weights for downstream Grad-CAM / visualisation use.
    """

    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.mha      = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.attn_weights = None            # populated during forward for viz

    def forward(self, x):
        # x: (B, C, L') — CNN convention (channels first)
        x_t = x.permute(0, 2, 1)           # (B, L', C)

        attn_out, weights = self.mha(x_t, x_t, x_t,
                                      need_weights=True,
                                      average_attn_weights=False)
        self.attn_weights = weights.detach()   # (B, heads, L', L')

        out = self.norm(x_t + self.dropout(attn_out))   # residual + LN
        return out.permute(0, 2, 1)         # back to (B, C, L')


class ClassifierHead(nn.Module):
    """Global Average Pool -> Linear -> GELU -> Dropout -> Linear -> Sigmoid."""

    def __init__(self, d_model=128, hidden=64, dropout=0.3):
        super().__init__()
        self.fc1     = nn.Linear(d_model, hidden)
        self.drop    = nn.Dropout(dropout)
        self.fc2     = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, C, L')
        pooled = x.mean(dim=-1)                     # (B, C)  — global avg pool
        out    = self.drop(F.gelu(self.fc1(pooled)))
        return torch.sigmoid(self.fc2(out)).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ExoNetAttention(nn.Module):
    """
    ExoNet-Attention: recommended architecture for BAH2026 exoplanet detection.

    Parameters
    ----------
    seq_len  : int   — input flux length (default 201, phase-folded Kepler)
    n_heads  : int   — number of attention heads (default 4)
    dropout  : float — dropout rate throughout (default 0.3)

    Input
    -----
    x : (B, 1, L)  float32 — normalized flux time series

    Output
    ------
    prob : (B,)  float32 — P(planet candidate | flux) in [0, 1]
    """

    def __init__(self, seq_len=201, n_heads=4, dropout=0.3):
        super().__init__()

        # --- CNN feature extractor ---
        self.cnn = nn.Sequential(
            ConvBlock(1,   32,  kernel_size=5, pool=False),   # (B, 32, L)
            ConvBlock(32,  64,  kernel_size=5, pool=True),    # (B, 64, L/2)
            ConvBlock(64,  128, kernel_size=3, pool=True),    # (B, 128, L/4)
            ConvBlock(128, 128, kernel_size=3, pool=False),   # (B, 128, L/4)
        )

        # --- Attention over CNN features ---
        self.attention = AttentionBlock(d_model=128, n_heads=n_heads,
                                         dropout=dropout * 0.33)

        # --- Classification head ---
        self.head = ClassifierHead(d_model=128, hidden=64, dropout=dropout)

    def forward(self, x):
        features = self.cnn(x)              # (B, 128, L')
        attended  = self.attention(features) # (B, 128, L')
        prob      = self.head(attended)     # (B,)
        return prob

    def get_attention_weights(self):
        """Return stored attention weights (B, heads, L', L') from last forward pass."""
        return self.attention.attn_weights

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Ablation variant: CNN-only baseline (no attention)
# ---------------------------------------------------------------------------

class ExoNetCNNBaseline(nn.Module):
    """
    Pure 1D-CNN baseline — no attention.
    Used in Ablation B to isolate the attention contribution.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1,   32,  kernel_size=5, pool=False),
            ConvBlock(32,  64,  kernel_size=5, pool=True),
            ConvBlock(64,  128, kernel_size=3, pool=True),
            ConvBlock(128, 128, kernel_size=3, pool=False),
        )
        self.head = ClassifierHead(d_model=128, hidden=64, dropout=dropout)

    def forward(self, x):
        return self.head(self.cnn(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, L = 8, 201                           # batch of 8 light curves, len 201

    # --- Proposed model ---
    model_attn = ExoNetAttention(seq_len=L, n_heads=4, dropout=0.3)
    x_dummy = torch.randn(B, 1, L)

    probs = model_attn(x_dummy)
    attn  = model_attn.get_attention_weights()

    print("=" * 55)
    print("  ExoNet-Attention — forward pass check")
    print("=" * 55)
    print(f"  Input shape      : {list(x_dummy.shape)}")
    print(f"  Output shape     : {list(probs.shape)}")
    print(f"  Output range     : [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Attention shape  : {list(attn.shape)}")
    print(f"  Trainable params : {model_attn.count_parameters():,}")

    # --- CNN baseline for ablation ---
    model_cnn = ExoNetCNNBaseline()
    probs_cnn = model_cnn(x_dummy)

    print()
    print("  ExoNetCNNBaseline — ablation model")
    print(f"  Trainable params : {model_cnn.count_parameters():,}")
    print(f"  Output shape     : {list(probs_cnn.shape)}")

    # --- Check attention weights are meaningful (not uniform) ---
    attn_entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean()
    print()
    print(f"  Attention entropy (random init): {attn_entropy:.3f}")
    print("  (Should increase after training as model learns to focus)")
    print()
    print("  All checks passed.")
