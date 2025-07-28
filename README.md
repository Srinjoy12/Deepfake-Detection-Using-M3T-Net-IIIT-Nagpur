# 🔍 Deepfake Detection Using M2T2-Net

An advanced AI-powered deepfake detection system developed at IIIT Nagpur using M2T2-Net (Multi-Modal Temporal Transformer Network).

![Deepfake Detector](https://img.shields.io/badge/AI-Deepfake%20Detection-blue)
![Framework](https://img.shields.io/badge/Framework-Next.js%20%2B%20FastAPI-green)
![Model](https://img.shields.io/badge/Model-M2T2--Net-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.7%25-brightgreen)

## 🌟 Features

- 🎯 **High Accuracy**: 99.7% detection accuracy using Vision Transformer
- ⚡ **Real-time Analysis**: Fast sliding window inference
- 🖥️ **GPU Accelerated**: Optimized for NVIDIA GPUs
- 📱 **Modern UI**: Beautiful Next.js frontend with Aurora background
- 🔒 **Privacy Focused**: Secure file processing
- 🚀 **Easy Deployment**: Ready for cloud deployment

## 🏗️ Architecture

### Frontend
- **Framework**: Next.js 15 with TypeScript
- **UI**: Tailwind CSS + shadcn/ui components
- **Features**: Drag-and-drop upload, real-time progress, Aurora background
- **Deployment**: Vercel-ready

### Backend
- **Framework**: FastAPI with Python 3.11+
- **Model**: M2T2-Net (Visual-only variant)
- **Processing**: Sliding window analysis (16-frame windows, 8-frame stride)
- **Features**: Real-time streaming, PDF report generation

### Model Architecture
1. **Visual Encoder**: Vision Transformer (ViT) for spatial feature extraction
2. **Temporal Encoder**: Transformer for temporal pattern analysis
3. **Classification Head**: Binary classification (authentic vs. deepfake)
4. **Sliding Window**: Analyzes video in overlapping segments

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and pnpm
- Python 3.11+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Srinjoy12/Deepfake-Detection-Using-M3T-Net-IIIT-Nagpur.git
   cd Deepfake-Detection-Using-M3T-Net-IIIT-Nagpur
   ```

2. **Setup Frontend**
   ```bash
   # Install dependencies
   pnpm install
   
   # Start development server
   pnpm dev
   ```

3. **Setup Backend**
   ```bash
   cd backend
   
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Start FastAPI server
   python main.py
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## 📖 Usage

1. **Upload Video**: Drag and drop or select a video file (MP4, MOV, AVI, MKV)
2. **Analyze**: Click "Begin Analysis" to start detection
3. **View Results**: See real-time analysis logs and final results
4. **Generate Report**: Download detailed PDF report with analysis

## 🎯 Model Performance

- **Dataset**: Trained on large-scale deepfake datasets
- **Accuracy**: 99.7% on test datasets
- **Speed**: ~2-5 seconds per video (GPU)
- **Supported Formats**: MP4, MOV, AVI, MKV
- **Max File Size**: 50MB (configurable)

## 🔧 Technical Details

### Dependencies
- **Frontend**: Next.js, React, Tailwind CSS, Framer Motion
- **Backend**: FastAPI, PyTorch, OpenCV, face_recognition
- **ML**: timm, torchvision, PIL, numpy

### Model Files
- `visual_only_best_model.pth` - Main detection model
- `final_m2t2_net_best.pth` - Alternative model variant

## 🚀 Deployment

### Free Deployment Options

1. **Hugging Face Spaces** (Recommended - Free GPU)
   - See `FREE_DEPLOYMENT.md` for detailed instructions
   - Includes free NVIDIA T4 GPU

2. **Vercel + Render.com**
   - Frontend on Vercel (free)
   - Backend on Render.com (free tier)

3. **Railway + Vercel**
   - See `DEPLOYMENT.md` for production deployment

## 📁 Project Structure

```
deepfake-detector/
├── app/                          # Next.js app directory
├── components/                   # React components
│   └── ui/                      # UI components (shadcn/ui)
├── backend/                     # FastAPI backend
│   ├── main.py                  # FastAPI server
│   ├── model.py                 # M2T2-Net model
│   ├── app.py                   # Gradio interface
│   ├── requirements.txt         # Python dependencies
│   └── *.pth                    # Model files
├── lib/                         # Utility functions
├── public/                      # Static assets
└── styles/                      # Global styles
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Context

This project was developed as part of research at **IIIT Nagpur** focusing on advanced deepfake detection techniques using transformer-based architectures.

## ⚠️ Disclaimer

This tool is designed for research and educational purposes. Results are not guaranteed to be 100% accurate and should be used for informational purposes only.

## 📞 Contact

- **Developer**: Srinjoy Roy
- **Institution**: IIIT Nagpur
- **GitHub**: [@Srinjoy12](https://github.com/Srinjoy12)

## 🙏 Acknowledgments

- IIIT Nagpur for research support
- Vision Transformer (ViT) architecture
- Open-source ML community
- Hugging Face for deployment platform

---

⭐ **Star this repository if you found it helpful!** 