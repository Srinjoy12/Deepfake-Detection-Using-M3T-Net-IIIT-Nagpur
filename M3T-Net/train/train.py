import os
import torch
from torch.utils.data import DataLoader
from models.m3t_net import M3TNet
from data.datasets import DeepfakeVideoDataset
import torch.nn as nn
import torch.optim as optim

# Configurations (edit as needed)
DATA_ROOT = './data'  # Path to your dataset root
LABEL_FILE = './data/labels.csv'  # Path to your label CSV
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_FRAMES = 10
FRAME_SIZE = 64
AUDIO_LEN = 480000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
train_dataset = DeepfakeVideoDataset(
    root_dir=DATA_ROOT,
    label_file=LABEL_FILE,
    num_frames=NUM_FRAMES,
    frame_size=FRAME_SIZE,
    audio_len=AUDIO_LEN
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = M3TNet(config=None)  # Pass config if needed
model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        frames, audio, label = batch[:3]  # Adjust if dataset returns more
        frames = frames.to(DEVICE)
        audio = audio.to(DEVICE)
        label = label.float().to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        class_out, _, _ = model(frames, audio)
        loss = criterion(class_out, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * frames.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

print("Training complete.")
