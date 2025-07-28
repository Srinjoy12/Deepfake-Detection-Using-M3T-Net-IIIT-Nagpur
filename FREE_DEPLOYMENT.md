# 🆓 Free Deepfake Detector Deployment Guide

## 🌟 **Option 1: Hugging Face Spaces** (Recommended)

### ✅ **Why Hugging Face Spaces?**
- **FREE GPU** (NVIDIA T4)
- Perfect for ML models
- Handles large files (your .pth models)
- No time limits
- Built-in Gradio interface
- Easy sharing and embedding

### 🚀 **Deploy to Hugging Face Spaces**

1. **Create Account**
   - Go to [huggingface.co](https://huggingface.co)
   - Sign up for free account

2. **Create New Space**
   - Click "Create new" → "Space"
   - Name: `deepfake-detector`
   - SDK: `Gradio`
   - Hardware: `T4 small` (free GPU!)
   - Visibility: `Public`

3. **Upload Files**
   ```bash
   # Clone your space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector
   cd deepfake-detector
   
   # Copy backend files
   cp -r ../backend/* .
   
   # Commit and push
   git add .
   git commit -m "Initial deployment"
   git push
   ```

4. **Files Structure for HF Spaces**
   ```
   deepfake-detector/
   ├── app.py                    # Main Gradio app
   ├── model.py                  # Your model code
   ├── requirements.txt          # Dependencies + gradio
   ├── README.md                 # HF Spaces metadata
   ├── visual_only_best_model.pth # Your model file
   └── final_m2t2_net_best.pth    # Your model file
   ```

5. **Your Space URL**
   - `https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector`
   - Automatic HTTPS and global CDN
   - Shareable and embeddable

---

## 🔄 **Option 2: Render.com Free Tier** (Backup)

### ⚠️ **Limitations**
- CPU only (slower inference)
- 750 hours/month free
- Sleeps after 15 min inactivity
- 512MB RAM limit

### 🚀 **Deploy to Render**

1. **Create Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Create Web Service**
   - Connect GitHub repository
   - Choose "Web Service"
   - Root Directory: `/backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python main.py`

3. **Environment Variables**
   - `PORT`: `10000`
   - `PYTHONUNBUFFERED`: `1`

---

## 🌐 **Frontend Deployment (Still Free!)**

### **Vercel** (Recommended)

1. **Deploy Frontend**
   ```bash
   # Update API URL for HF Spaces
   # In next.config.mjs, set:
   NEXT_PUBLIC_API_URL: 'https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector'
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Import GitHub repo
   - Set environment variable:
     - `NEXT_PUBLIC_API_URL`: Your HF Spaces URL

---

## 📊 **Comparison: Free Options**

| Feature | Hugging Face Spaces | Render Free |
|---------|-------------------|-------------|
| **Cost** | 100% Free | 100% Free |
| **GPU** | ✅ NVIDIA T4 | ❌ CPU only |
| **Model Size** | ✅ Large files OK | ⚠️ Limited |
| **Uptime** | ✅ Always on | ⚠️ Sleeps |
| **Speed** | ⚡ Fast (GPU) | 🐌 Slow (CPU) |
| **Setup** | 🟢 Easy | 🟡 Medium |

---

## 🎯 **Recommended Approach**

### **Primary: Hugging Face Spaces**
```bash
# 1. Create HF Space with GPU
# 2. Upload your model files
# 3. Deploy with Gradio interface
# 4. Get free GPU inference!
```

### **Frontend: Vercel**
```bash
# 1. Deploy Next.js to Vercel (free)
# 2. Point to HF Spaces API
# 3. Enjoy free full-stack deployment!
```

---

## 🔧 **Setup Commands**

### **For Hugging Face Spaces:**
```bash
# In your backend directory
cd backend

# Test locally first
python app.py

# Then upload to HF Spaces
git add .
git commit -m "Deploy to HF Spaces"
git push
```

### **Update Frontend:**
```bash
# Update API URL in next.config.mjs
NEXT_PUBLIC_API_URL: 'https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector'

# Deploy to Vercel
vercel --prod
```

---

## 💡 **Pro Tips**

1. **Model Files**: HF Spaces handles large files automatically
2. **GPU Usage**: T4 GPU is perfect for your PyTorch models
3. **Sharing**: HF Spaces URLs are easily shareable
4. **Monitoring**: Built-in logs and metrics
5. **Community**: Great for showcasing ML projects

---

## 🚨 **Troubleshooting**

### **HF Spaces Issues:**
- Check `app.py` is the main file
- Verify `requirements.txt` includes `gradio`
- Ensure model files are uploaded
- Check logs in HF Spaces interface

### **Render Issues:**
- Increase build timeout if needed
- Check memory usage (512MB limit)
- Verify Python version compatibility

---

## 💰 **Total Cost: $0/month** 🎉

- **Backend**: Hugging Face Spaces (Free GPU)
- **Frontend**: Vercel (Free tier)
- **Domain**: Free subdomains included
- **SSL**: Automatic HTTPS

**Perfect for:** Portfolio projects, demos, research, learning

---

Ready to deploy for free? Start with Hugging Face Spaces! 🚀 