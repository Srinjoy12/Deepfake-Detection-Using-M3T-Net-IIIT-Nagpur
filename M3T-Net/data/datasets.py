import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchaudio

class DeepfakeVideoDataset(Dataset):
    """
    Multimodal dataset for deepfake detection.
    Loads video frames, audio waveform, labels, and optional segmentation masks.
    Searches for video files in all relevant subfolders.
    Handles missing audio by using zeros.
    """
    def __init__(self, root_dir, label_file, num_frames=10, frame_size=64, audio_len=480000, transform=None, mask_dir=None):
        """
        root_dir: path to dataset root
        label_file: path to CSV with video_id,label
        num_frames: number of frames to sample per video
        frame_size: resize frames to (frame_size, frame_size)
        audio_len: length of audio waveform to extract (in samples)
        transform: optional transform for frames
        mask_dir: path to segmentation masks (optional)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.audio_len = audio_len
        self.mask_dir = mask_dir
        self.subfolders = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']

        # Load labels
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f.readlines()[1:]:
                video_id, label = line.strip().split(',')
                self.samples.append((video_id, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label = self.samples[idx]
        # Search for the video in all subfolders
        video_path = None
        for subfolder in self.subfolders:
            candidate = os.path.join(self.root_dir, subfolder, f'{video_id}.mp4')
            if os.path.exists(candidate):
                video_path = candidate
                break
        if video_path is None:
            raise FileNotFoundError(f"Video file for {video_id} not found in any subfolder.")

        # --- Load frames ---
        frames = self._sample_frames(video_path)
        if self.transform:
            frames = [self.transform(f) for f in frames]
        frames = np.stack(frames)  # (T, H, W, C)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.  # (T, C, H, W)

        # --- Load audio (handle missing audio) ---
        audio_path = os.path.join(self.root_dir, 'audio', f'{video_id}.wav')
        if os.path.exists(audio_path):
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.mean(dim=0)  # mono
            if waveform.shape[0] > self.audio_len:
                waveform = waveform[:self.audio_len]
            else:
                pad = self.audio_len - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = torch.zeros(self.audio_len)

        # --- Load mask (optional) ---
        mask = None
        if self.mask_dir:
            mask = []
            for i in range(self.num_frames):
                mask_path = os.path.join(self.mask_dir, video_id, f'{i+1:05d}.png')
                if os.path.exists(mask_path):
                    m = cv2.imread(mask_path, 0)
                    m = cv2.resize(m, (self.frame_size, self.frame_size))
                    mask.append(m)
                else:
                    mask.append(np.zeros((self.frame_size, self.frame_size)))
            mask = np.stack(mask)
            mask = torch.from_numpy(mask).unsqueeze(1).float() / 255.  # (T, 1, H, W)

        return {
            'frames': frames,         # (T, C, H, W)
            'audio': waveform,        # (audio_len,)
            'label': torch.tensor(label, dtype=torch.float32),
            'mask': mask              # (T, 1, H, W) or None
        }

    def _sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, frame_count-1, self.num_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frames.append(frame)
        cap.release()
        return frames
