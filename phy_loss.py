import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsLoss(nn.Module):
    def __init__(self, sym_weight=0.05, depth_weight=0.05):
        super().__init__()
        self.sym_weight = sym_weight
        self.depth_weight = depth_weight

    def forward(self, x, attn_weights):
        """
        x: (B, 1, L) input flux
        attn_weights: (B, heads, L', L') - handles L' != L automatically
        """
        # Automatically resize attention weights to match input length (L) if they differ
        if attn_weights.shape[-1] != x.shape[-1]:
            attn_weights = F.interpolate(attn_weights, size=(x.shape[-1], x.shape[-1]), 
                                         mode='bilinear', align_corners=False)

        # Average over heads to get a single (B, L, L) map
        attn = attn_weights.mean(dim=1) 
        
        # Collapse to (B, L) for physics calculations[cite: 3]
        attn_1d = attn.mean(dim=1) 
        
        # 1. Symmetry Loss with robust slicing[cite: 3]
        peak_idx = torch.argmax(attn_1d, dim=-1)
        sym_loss = 0.0
        for i in range(x.shape[0]):
            p = peak_idx[i]
            left_side = attn_1d[i, :p]
            right_side = attn_1d[i, p:]
            min_len = min(len(left_side), len(right_side))
            if min_len > 0:
                sym_loss += F.mse_loss(left_side[:min_len], torch.flip(right_side[:min_len], [-1]))
        
        sym_loss = sym_loss / x.shape[0] if x.shape[0] > 0 else 0.0
        
        # 2. Depth/Flat-bottom Consistency[cite: 3]
        flat_loss = torch.var(x.squeeze(1) * attn_1d, dim=-1).mean()
        
        return self.sym_weight * sym_loss + self.depth_weight * flat_loss