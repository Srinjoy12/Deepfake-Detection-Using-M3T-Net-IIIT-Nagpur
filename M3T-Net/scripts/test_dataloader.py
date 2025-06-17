import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from data.datasets import DeepfakeVideoDataset

if __name__ == "__main__":
    dataset = DeepfakeVideoDataset(
        root_dir='/Volumes/Sumit_HD/dataset/archive (1)',
        label_file='/Volumes/Sumit_HD/dataset/archive (1)/labels.csv',
        num_frames=10,
        frame_size=64,
        audio_len=480000,
        mask_dir=None
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    for batch in loader:
        frames = batch['frames']      # (B, T, C, H, W)
        audio = batch['audio']        # (B, audio_len)
        label = batch['label']        # (B,)
        mask = batch['mask']          # (B, T, 1, H, W) or None
        print(f"frames shape: {frames.shape}")
        print(f"audio shape: {audio.shape}")
        print(f"label shape: {label.shape}")
        print(f"mask shape: {mask.shape if mask is not None else None}")
        break 