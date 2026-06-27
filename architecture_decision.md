# ML Architecture Decision — Exoplanet Detection from Light Curves
## BAH2026 | Team: Shreya · Ananya · Sara · Tejeswin

---

## Problem framing

**Input**: 1D flux time series of length L (typically 200–2000 timesteps after preprocessing)  
**Output**: Binary classification — Planet Candidate (PC=1) vs Non-Planet (NP=0)  
**Key difficulty**: Class imbalance (~10% PC in Kepler TCE catalog), transit signal is ~0.01–1% of total flux, noise dominated by stellar variability

---

## Architectures surveyed

### 1. 1D-CNN (Convolutional Neural Network)

**How it works**: Stacked 1D convolutions detect local patterns (transit box shape) at multiple scales via kernel sizes and pooling.

**Pros**:
- Translationally equivariant — detects transit regardless of phase offset
- Very fast to train (no sequential bottleneck)
- Proven on Kepler: Shallue & Vanderburg (2018) "Identifying Exoplanets with Deep Learning" used pure CNN, 97.5% AUC
- Easy to visualize: Grad-CAM works naturally on 1D conv filters

**Cons**:
- Fixed receptive field — misses long-range dependencies (e.g. transit timing variations)
- No explicit attention over which time steps matter
- Requires careful kernel size selection (transit duration varies 1–10 hrs)

**Compute cost**: Low. ~500k params, trains in minutes on CPU.

---

### 2. Bi-LSTM (Bidirectional Long Short-Term Memory)

**How it works**: Sequential model processing flux forward and backward, maintaining hidden state to capture temporal dependencies.

**Pros**:
- Captures long-range temporal correlations in stellar variability
- Bidirectional: uses future context (useful for secondary eclipse detection)

**Cons**:
- Slow to train on long sequences (O(L) sequential computation, no parallelism)
- Vanishing gradient problem on very long light curves (L > 1000)
- Overfits easily on small datasets
- Harder to interpret than CNN
- Not competitive vs CNN+Attention in recent benchmarks (Malik et al. 2022)

**Compute cost**: High for L > 500. 3–5× slower than CNN per epoch.

---

### 3. Transformer / Self-Attention

**How it works**: Multi-head self-attention computes pairwise relationships between all timesteps simultaneously.

**Pros**:
- Global receptive field from layer 1 — captures transit timing variations
- Attention weights directly interpretable: show which timesteps drive prediction
- State-of-the-art on time series classification benchmarks (2022–2024)

**Cons**:
- O(L²) memory complexity — expensive for raw long-cadence light curves
- Requires positional encoding (non-trivial for irregular Kepler timestamps)
- Needs large datasets to train from scratch; benefits from pretraining
- Overkill for L < 1000 after phase-folding

**Compute cost**: Moderate–High. Feasible with L ≤ 512 on CPU.

---

### 4. 1D-CNN + Multi-Head Attention (HYBRID) ✓ RECOMMENDED

**How it works**:
1. 1D-CNN layers extract local feature maps (transit shape, ingress/egress edges)
2. Multi-Head Attention over CNN feature maps — attends globally across timesteps
3. Global Average Pooling → Dense head → sigmoid output

**Why this wins**:
- CNN handles the O(L) local feature extraction cheaply
- Attention then operates on the compressed feature space (L' << L), so O(L'²) is tractable
- Best of both: local pattern detection + global temporal reasoning
- Attention weights = built-in explainability (Sara's Grad-CAM section)
- Closest to Dattilo et al. (2019) "Identifying Exoplanets with Deep Learning II" which used CNN+RNN; we modernize with attention
- Ablation is clean: CNN-only vs CNN+Attention directly measures attention's contribution

**Compute cost**: Low–Moderate. ~1.2M params, trains in 20–40 min on CPU.

---

## Architecture comparison matrix

| Property | 1D-CNN | Bi-LSTM | Transformer | **CNN+Attention** |
|---|---|---|---|---|
| Local pattern detection | ✓✓ | ✓ | ✗ (indirect) | **✓✓** |
| Long-range dependencies | ✗ | ✓✓ | ✓✓ | **✓✓** |
| Training speed (CPU) | Fast | Slow | Moderate | **Fast** |
| Interpretability | Grad-CAM | Poor | Attention maps | **Grad-CAM + Attention** |
| Param count (est.) | ~500k | ~800k | ~2M | **~1.2M** |
| Data efficiency | High | Low | Low | **High** |
| Overfitting risk | Low | High | High | **Low-Moderate** |
| Proven on exoplanets | ✓ (S&V 2018) | Partial | No | **Builds on S&V** |
| Ablation clarity | Baseline | Poor | Standalone | **CNN as ablation** |

---

## Recommended architecture: ExoNet-Attention

```
Input flux [B, 1, L]
    │
    ├─ Conv1D(1→32, k=5) → BN → ReLU
    ├─ Conv1D(32→64, k=5) → BN → ReLU → MaxPool(2)
    ├─ Conv1D(64→128, k=3) → BN → ReLU → MaxPool(2)
    ├─ Conv1D(128→128, k=3) → BN → ReLU
    │         [B, 128, L']   (L' ≈ L/4)
    │
    ├─ Permute → [B, L', 128]
    ├─ MultiHeadAttention(heads=4, d_model=128)
    ├─ Residual connection + LayerNorm
    │
    ├─ Global Average Pooling → [B, 128]
    ├─ Linear(128→64) → GELU → Dropout(0.3)
    ├─ Linear(64→1) → Sigmoid
    │
Output: P(planet | flux) ∈ [0,1]
```

---

## Ablation experiment plan

| Experiment | Change | Metric to compare |
|---|---|---|
| Baseline A | BLS only (Ananya's classical) | AUC, F1 |
| Baseline B | 1D-CNN, no attention | AUC, F1, Grad-CAM |
| **Proposed** | 1D-CNN + MHA | AUC, F1, attention entropy |
| Ablation 1 | Remove residual connection | AUC |
| Ablation 2 | 2 heads vs 4 vs 8 | AUC, training time |
| Ablation 3 | No physics loss (Sara's) vs with | Precision on edge-case transits |

---

## Hyperparameter search plan (Phase 3)

Using Optuna with 30 trials, TPE sampler:

```python
# Search space
lr          ∈ [1e-4, 1e-2]   log-uniform
dropout     ∈ [0.1, 0.5]
n_heads     ∈ {2, 4, 8}
kernel_sizes ∈ {(5,5,3,3), (7,5,3,3), (5,3,3,3)}
weight_decay ∈ [1e-5, 1e-3]  log-uniform
```

Stopping criterion: Early stopping on val AUC, patience=10 epochs.

---

## Dataset + input spec (for Shreya's pipeline)

**Expected input from Shreya:**
```
shape:  (N_samples, 1, L)   where L = 201 (phase-folded, normalized)
dtype:  float32
range:  mean ≈ 1.0, normalized around transit minimum
files:  train.npz, val.npz, test.npz
keys:   'flux', 'labels', 'kic_id', 'period', 'duration'
```

**Labels (from Ananya's TCE catalog work):**
```
1 = Planet Candidate (PC)
0 = Non-Planet (AFP + NTP + FP)
```

**Class imbalance handling:**
- Weighted CrossEntropy: pos_weight = N_neg / N_pos
- Or: oversample PC class during DataLoader

---

*Document version: Phase 1 | Author: Tejeswin R | BAH2026*
