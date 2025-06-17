import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Binary classification head for real/fake prediction.
    Input: (B, T, D) or (B, D)
    Output: (B, 1) or (B, T, 1)
    """
    def __init__(self, in_dim, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        if self.temporal:
            # x: (B, T, D) -> (B, T, 1)
            return torch.sigmoid(self.fc(x))
        else:
            # x: (B, D) or (B, 1, D)
            if x.dim() == 3:
                x = x.mean(dim=1)  # Pool over time
            return torch.sigmoid(self.fc(x))  # (B, 1)

class SegmentationHead(nn.Module):
    """
    Frame-level segmentation head for partial deepfake detection.
    Input: (B, T, D)
    Output: (B, T, 1) (mask per frame)
    """
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        return torch.sigmoid(self.fc(x))  # (B, T, 1)

class FairnessHead(nn.Module):
    """
    Optional fairness/bias score head.
    Input: (B, D) or (B, T, D)
    Output: (B, 1) or (B, T, 1)
    """
    def __init__(self, in_dim, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        if self.temporal:
            return torch.sigmoid(self.fc(x))  # (B, T, 1)
        else:
            if x.dim() == 3:
                x = x.mean(dim=1)
            return torch.sigmoid(self.fc(x))  # (B, 1)
