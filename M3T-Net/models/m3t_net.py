import torch
import torch.nn as nn
from .dp_mask import DPMask
from .m2tr import M2TR
from .audio_encoder import AudioEncoder
from .fusion import FusionTransformer
from .temporal import TimeseriesTransformer
from .heads import ClassificationHead, SegmentationHead, FairnessHead

class M3TNet(nn.Module):
    """
    Main M3T-Net model integrating all modules for multimodal deepfake detection.
    Inputs:
        frames: (B, C, H, W) or (B, T, C, H, W)  # video frames
        audio: (B, T_audio)  # audio waveform
    Outputs:
        class_out: (B, 1)  # real/fake
        seg_out: (B, T, 1)  # frame-level mask
        fairness_out: (B, 1) or (B, T, 1)  # optional
    """
    def __init__(self, config):
        super().__init__()
        self.dp_mask = DPMask()
        self.visual_stream = M2TR(in_channels=6, embed_dim=128, num_heads=2, num_layers=1, patch_size=16)
        self.audio_encoder = AudioEncoder(model_name='openai/whisper-tiny')
        self.fusion = FusionTransformer(visual_dim=128, audio_dim=384, fusion_dim=256, num_heads=2, num_layers=1)
        self.temporal = TimeseriesTransformer(embed_dim=256, num_heads=2, num_layers=1)
        self.class_head = ClassificationHead(in_dim=256)
        self.seg_head = SegmentationHead(in_dim=256)
        self.fairness_head = FairnessHead(in_dim=256)

    def forward(self, frames, audio):
        # frames: (B, T, C, H, W)
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        dp_masked = self.dp_mask(frames)  # (B*T, 6, H, W)
        visual_feat = self.visual_stream(dp_masked)  # (B*T, N_v, 128)
        N_v = visual_feat.shape[1]
        visual_feat = visual_feat.view(B, T, N_v, 128)
        # Pool visual features per frame (mean over patches)
        visual_feat = visual_feat.mean(dim=2)  # (B, T, 128)
        # Audio embedding (B, 384)
        audio_emb, _ = self.audio_encoder(audio)
        # Repeat audio embedding for each frame
        audio_feat = audio_emb.unsqueeze(1).repeat(1, T, 1)  # (B, T, 384)
        # Fuse modalities per frame
        fused = self.fusion(visual_feat, audio_feat)  # (B, T, 256)
        # Temporal modeling
        temporal_out = self.temporal(fused)  # (B, T, 256)
        # Output heads
        class_out = self.class_head(temporal_out)  # (B, 1)
        seg_out = self.seg_head(temporal_out)      # (B, T, 1)
        fairness_out = self.fairness_head(temporal_out)  # (B, 1)
        return class_out, seg_out, fairness_out
