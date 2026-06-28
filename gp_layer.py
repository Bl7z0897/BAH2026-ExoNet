import torch
import torch.nn as nn

class GPDetrendLayer(nn.Module):
    """
    Simulates GP detrending by learning a smoothing kernel to 
    remove low-frequency stellar variability (the 'trend').
    """
    def __init__(self, kernel_size=31):
        super().__init__()
        # Use a depthwise convolution as a trainable smoothing kernel
        self.smoother = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                                  padding=kernel_size//2, bias=False)
        # Initialize as a Gaussian-like moving average
        nn.init.constant_(self.smoother.weight, 1.0 / kernel_size)

    def forward(self, x):
        # x: (B, 1, L)
        # Identify the low-frequency trend
        trend = self.smoother(x)
        # Return the residual (signal - trend)
        return x - trend