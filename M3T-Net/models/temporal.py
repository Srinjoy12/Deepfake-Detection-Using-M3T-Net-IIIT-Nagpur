import torch
import torch.nn as nn

class TimeseriesTransformer(nn.Module):
    """
    Transformer for temporal modeling of fused multimodal features.
    Input: (B, T, D) where T is the number of time steps (frames)
    Output: (B, T, D)
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, T, D)
        # Transformer expects (T, B, D)
        x = x.permute(1, 0, 2)
        out = self.transformer(x)
        out = out.permute(1, 0, 2)  # (B, T, D)
        return out
