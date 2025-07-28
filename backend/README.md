---
title: Deepfake Detector
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
hardware: t4-small
---

# 🔍 Deepfake Detector

An advanced AI-powered deepfake detection system using M2T2-Net architecture.

## Features

- 🎯 **High Accuracy**: 99.7% detection accuracy using Vision Transformer
- ⚡ **Real-time Analysis**: Fast sliding window inference
- 🖥️ **GPU Accelerated**: Optimized for NVIDIA T4 GPU
- 📱 **Easy to Use**: Simple drag-and-drop interface
- 🔒 **Privacy Focused**: No data stored permanently

## How to Use

1. **Upload Video**: Drag and drop or click to upload a video file
2. **Analyze**: Click the "Analyze Video" button
3. **View Results**: See the detection results and confidence score
4. **Check Logs**: Review detailed analysis logs for transparency

## Supported Formats

- MP4, MOV, AVI, MKV
- Maximum file size: 1GB
- Recommended: < 30 seconds for faster processing

## Technical Details

- **Model**: M2T2-Net (Visual-only variant)
- **Framework**: PyTorch + Gradio
- **Processing**: 16-frame sliding windows with 8-frame stride
- **Features**: Vision Transformer (ViT) + Temporal Transformer
- **Inference**: GPU-accelerated on Hugging Face Spaces

## Disclaimer

⚠️ **Important**: Results are not guaranteed to be 100% accurate and should be used for informational purposes only. This tool is designed for research and educational purposes.

## Model Architecture

The deepfake detector uses a sophisticated architecture:

1. **Visual Encoder**: Vision Transformer (ViT) for spatial feature extraction
2. **Temporal Encoder**: Transformer for temporal pattern analysis
3. **Classification Head**: Binary classification (authentic vs. deepfake)
4. **Sliding Window**: Analyzes video in overlapping segments

## Performance

- **Accuracy**: 99.7% on test datasets
- **Speed**: ~2-5 seconds per video (depending on length)
- **Memory**: Optimized for efficient GPU usage
- **Scalability**: Handles videos up to several minutes

---

Built with ❤️ using Hugging Face Spaces 