import torch
import torch.nn as nn
import numpy as np
import torch.fft

class DPMask(nn.Module):
    """
    DPMask: Preprocessing module to amplify subtle forgery artifacts in spatial and frequency domains.
    Input: RGB image tensor (B, C, H, W)
    Output: Concatenated tensor of RGB and frequency domain (B, 2*C, H, W)
    """
    def __init__(self):
        super(DPMask, self).__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        # Compute FFT on each channel
        freq = torch.fft.fft2(x)
        freq = torch.abs(torch.fft.fftshift(freq))
        # Normalize frequency domain
        freq = (freq - freq.mean(dim=(-2, -1), keepdim=True)) / (freq.std(dim=(-2, -1), keepdim=True) + 1e-6)
        # Concatenate spatial and frequency domain
        out = torch.cat([x, freq], dim=1)
        return out
