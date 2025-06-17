import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    """
    Attention-based fusion transformer for combining visual and audio modalities.
    Inputs:
        visual_feat: (B, N_v, D_v)
        audio_feat: (B, N_a, D_a) or (B, D_a)
    Output:
        fused_feat: (B, N_v + N_a, D_fusion)
    """
    def __init__(self, visual_dim, audio_dim, fusion_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, visual_feat, audio_feat):
        # visual_feat: (B, N_v, D_v)
        # audio_feat: (B, N_a, D_a) or (B, D_a)
        B = visual_feat.size(0)
        visual_proj = self.visual_proj(visual_feat)  # (B, N_v, fusion_dim)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)  # (B, 1, D_a)
        audio_proj = self.audio_proj(audio_feat)   # (B, N_a, fusion_dim)
        # Concatenate along sequence dimension
        fused = torch.cat([visual_proj, audio_proj], dim=1)  # (B, N_v + N_a, fusion_dim)
        # Transformer expects (N, B, D)
        fused = fused.permute(1, 0, 2)
        fused_out = self.transformer(fused)
        fused_out = fused_out.permute(1, 0, 2)  # (B, N_v + N_a, fusion_dim)
        return fused_out
