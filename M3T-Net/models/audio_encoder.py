import torch
import torch.nn as nn
import numpy as np
from transformers import WhisperProcessor, WhisperModel

class AudioEncoder(nn.Module):
    """
    AudioEncoder using Whisper for audio embedding.
    Input: waveform tensor (B, T) (single-channel, 16kHz recommended)
    Output: embeddings (B, D), transcript (None for now)
    """
    def __init__(self, model_name='openai/whisper-tiny'):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.eval()

    def forward(self, audio_waveform, sampling_rate=16000):
        # audio_waveform: (B, T) torch tensor
        if isinstance(audio_waveform, torch.Tensor):
            audio_waveform = audio_waveform.cpu().numpy()
        if isinstance(audio_waveform, np.ndarray):
            audio_waveform = [a for a in audio_waveform]  # List of 1D arrays
        inputs = self.processor(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model.encoder(inputs.input_features)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # (B, D)
        return embeddings, None
