import torch
from torch.utils.data import DataLoader
from models.m3t_net import M3TNet
from data.datasets import DeepfakeVideoDataset
import torch.nn as nn
from train.utils import compute_accuracy, compute_auc
import numpy as np

# Configurations (edit as needed)
DATA_ROOT = './data'  # Path to your dataset root
LABEL_FILE = './data/labels.csv'  # Path to your label CSV
BATCH_SIZE = 2
CHECKPOINT_PATH = './m3tnet_checkpoint.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
eval_dataset = DeepfakeVideoDataset(
    root_dir=DATA_ROOT,
    label_file=LABEL_FILE,
    num_frames=10,
    frame_size=64,
    audio_len=480000
)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = M3TNet(config=None)
model = model.to(DEVICE)

# Load checkpoint (if available)
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
except Exception as e:
    print(f"Could not load checkpoint: {e}")

model.eval()
criterion = nn.BCELoss()
all_labels = []
all_preds = []
total_loss = 0.0

with torch.no_grad():
    for batch in eval_loader:
        frames, audio, label = batch[:3]
        frames = frames.to(DEVICE)
        audio = audio.to(DEVICE)
        label = label.float().to(DEVICE).unsqueeze(1)
        class_out, _, _ = model(frames, audio)
        loss = criterion(class_out, label)
        total_loss += loss.item() * frames.size(0)
        all_labels.extend(label.cpu().numpy().flatten())
        all_preds.extend(class_out.cpu().numpy().flatten())

avg_loss = total_loss / len(eval_loader.dataset)
acc = compute_accuracy(np.array(all_labels), np.array(all_preds))
auc = compute_auc(np.array(all_labels), np.array(all_preds))

print(f"Eval Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")
