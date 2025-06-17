import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dp_mask import DPMask
from models.m2tr import M2TR
from models.audio_encoder import AudioEncoder
from models.fusion import FusionTransformer
from models.temporal import TimeseriesTransformer
from models.heads import ClassificationHead, SegmentationHead, FairnessHead
from models.m3t_net import M3TNet

class DummyConfig:
    pass

if __name__ == "__main__":
    # Simulate a batch of RGB images (B, C, H, W)
    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W)

    # Initialize modules
    dp_mask = DPMask()
    m2tr = M2TR(in_channels=6, embed_dim=128, num_heads=2, num_layers=1, patch_size=16)

    # DPMask preprocessing
    x_dp = dp_mask(x)
    print(f"DPMask output shape: {x_dp.shape}")  # Expect (B, 6, H, W)

    # M2TR feature extraction
    vis_feat = m2tr(x_dp)
    print(f"M2TR output shape: {vis_feat.shape}")  # Expect (B, N, D)

    # AudioEncoder test
    print("\nTesting AudioEncoder...")
    B, T_audio = 2, 480000  # 30 seconds of audio at 16kHz
    audio = torch.randn(B, T_audio)
    audio_encoder = AudioEncoder(model_name='openai/whisper-tiny')
    embeddings, transcripts = audio_encoder(audio, sampling_rate=16000)
    print(f"AudioEncoder embeddings shape: {embeddings.shape}")
    print(f"AudioEncoder transcripts: {transcripts}")

    # FusionTransformer test
    print("\nTesting FusionTransformer...")
    B, N_v, D_v = 2, 4, 128
    N_a, D_a = 1, embeddings.shape[1]
    visual_feat = torch.randn(B, N_v, D_v)
    audio_feat = embeddings.unsqueeze(1)  # (B, 1, D_a)
    fusion = FusionTransformer(visual_dim=D_v, audio_dim=D_a, fusion_dim=256, num_heads=2, num_layers=1)
    fused = fusion(visual_feat, audio_feat)
    print(f"FusionTransformer output shape: {fused.shape}")  # Expect (B, N_v + N_a, fusion_dim)

    # TimeseriesTransformer test
    print("\nTesting TimeseriesTransformer...")
    B, T, D = 2, 5, 256
    timeseries = torch.randn(B, T, D)
    temporal = TimeseriesTransformer(embed_dim=256, num_heads=2, num_layers=1)
    temporal_out = temporal(timeseries)
    print(f"TimeseriesTransformer output shape: {temporal_out.shape}")  # Expect (B, T, D)

    # Output heads test
    print("\nTesting Output Heads...")
    class_head = ClassificationHead(in_dim=256)
    seg_head = SegmentationHead(in_dim=256)
    fairness_head = FairnessHead(in_dim=256)
    print(f"ClassificationHead output shape: {class_head(temporal_out).shape}")
    print(f"SegmentationHead output shape: {seg_head(temporal_out).shape}")
    print(f"FairnessHead output shape: {fairness_head(temporal_out).shape}")

    # Full M3TNet test
    print("\nTesting full M3TNet model...")
    B, T, C, H, W = 2, 5, 3, 64, 64
    frames = torch.randn(B, T, C, H, W)
    audio = torch.randn(B, 480000)  # 30 seconds of audio
    config = DummyConfig()
    model = M3TNet(config)
    class_out, seg_out, fairness_out = model(frames, audio)
    print(f"M3TNet class_out shape: {class_out.shape}")
    print(f"M3TNet seg_out shape: {seg_out.shape}")
    print(f"M3TNet fairness_out shape: {fairness_out.shape}")
