import torch
import torch.nn as nn
import torch.nn.functional as F

class M2TR(nn.Module):
    """
    Multi-scale Transformer for visual stream (M2TR)
    Input: (B, 2*C, H, W) from DPMask
    Output: Visual feature embeddings (B, N, D)
    """
    def __init__(self, in_channels=6, embed_dim=256, num_heads=4, num_layers=2, patch_size=16):
        super(M2TR, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Multi-scale feature extraction (simple example: 2 convs)
        self.conv1 = nn.Conv2d(in_channels, embed_dim//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1)
        # Flatten patches
        self.proj = nn.Linear((patch_size**2)*embed_dim, embed_dim)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, 2*C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # (B, embed_dim, H/2, W/2)
        B, C, H, W = x.shape
        # Divide into non-overlapping patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0,2,1,3,4).contiguous().view(B, -1, C*self.patch_size*self.patch_size)  # (B, N, patch_dim)
        # Project to embedding
        patches = self.proj(patches)  # (B, N, embed_dim)
        # Transformer expects (N, B, D)
        patches = patches.permute(1,0,2)
        out = self.transformer(patches)
        out = out.permute(1,0,2)  # (B, N, D)
        return out
